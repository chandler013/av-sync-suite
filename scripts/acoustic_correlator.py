"""
Acoustic Correlator & Clock Drift Analyzer.
Used for same-language source synchronization via 100Hz RMS energy envelopes.
"""

import os
import subprocess
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import soundfile as sf
from scipy import signal
from .utils import FFMPEG


def extract_reference_audio(
    input_media: str,
    output_wav: str,
    stream_index: int = 0,
    channels: int = 2,
    sample_rate: int = 16000,
) -> None:
    """
    Extracts 16kHz mono reference audio for correlation.
    For 5.1/5.0 multi-channel audio, extracts ONLY the Center channel (Channel 2)
    to isolate speech transients and prevent surround phase cancellation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    audio_filter = "pan=mono|c0=FC" if channels >= 5 else "pan=mono|c0=0.5*c0+0.5*c1"

    cmd = [
        FFMPEG, "-y",
        "-i", input_media,
        "-vn",
        "-map", f"0:a:{stream_index}",
        "-af", audio_filter,
        "-ar", str(sample_rate),
        output_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def compute_envelope(wav_path: str, hop_samples: int = 160) -> np.ndarray:
    """
    Computes 100Hz Block RMS energy envelope with 50ms Gaussian smoothing.
    Robust against mastering EQ, dynamic range compression, and lossy codec phase alterations.
    """
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    num_blocks = len(audio) // hop_samples
    if num_blocks == 0:
        return np.array([], dtype=np.float32)

    blocks = audio[: num_blocks * hop_samples].reshape((num_blocks, hop_samples))
    rms = np.sqrt(np.mean(blocks**2, axis=1) + 1e-8)

    # 5-point Gaussian smoothing filter
    kernel = np.array([0.061, 0.242, 0.383, 0.242, 0.061], dtype=np.float32)
    return signal.convolve(rms, kernel, mode="same")


def find_acoustic_matches(
    ref_target_wav: str,
    ref_donor_wav: str,
    step_sec: float = 2.0,
    win_sec: float = 12.0,
    search_window: float = 30.0,
    env_rate: int = 100,
) -> List[Dict[str, float]]:
    """
    Correlates RMS energy envelopes across checkpoints.
    Returns list of {'t', 'offset', 'score'}.
    """
    env_t = compute_envelope(ref_target_wav, hop_samples=16000 // env_rate)
    env_d = compute_envelope(ref_donor_wav, hop_samples=16000 // env_rate)

    n_t, n_d = len(env_t), len(env_d)
    win_len = int(win_sec * env_rate)
    search_len = int(search_window * env_rate)

    if n_t < win_len or n_d < win_len:
        return []

    checkpoints = np.arange(0.0, (n_t / env_rate) - win_sec, step_sec)
    matches: List[Dict[str, float]] = []
    expected_offset_samples = 0

    for t_sec in checkpoints:
        idx_t = int(t_sec * env_rate)
        chunk_t = env_t[idx_t : idx_t + win_len]
        std_t = np.std(chunk_t)
        if std_t < 1e-4:
            continue

        chunk_t_norm = (chunk_t - np.mean(chunk_t)) / (std_t + 1e-4)

        d_center = idx_t + expected_offset_samples
        d_start = max(0, d_center - search_len)
        d_end = min(n_d - win_len, d_center + search_len)
        if d_end <= d_start:
            continue

        slice_d = env_d[d_start : d_end + win_len]
        # Sliding correlation via numpy correlate
        corr = signal.correlate(slice_d, chunk_t_norm, mode="valid")
        norm_factor = signal.convolve(slice_d**2, np.ones(win_len), mode="valid")
        scores = corr / (np.sqrt(norm_factor * win_len) + 1e-4)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_d_idx = d_start + best_idx
        offset_sec = (best_d_idx - idx_t) / env_rate

        if best_score >= 0.40:
            matches.append({"t": float(t_sec), "offset": float(offset_sec), "score": best_score})
            expected_offset_samples = int(offset_sec * env_rate)

    return matches


def analyze_clock_drift(matches: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Fits linear regression on correlation anchors to diagnose clock drift or framerate mismatch.
    Returns:
        {'slope': slope, 'base_offset': base, 'r_squared': r2, 'diagnosis': str}
    """
    if len(matches) < 5:
        return {"slope": 0.0, "base_offset": 0.0, "r_squared": 0.0, "diagnosis": "Insufficient data"}

    t_vals = np.array([m["t"] for m in matches])
    off_vals = np.array([m["offset"] for m in matches])

    # Robust fit (ignore outer 5% outliers)
    diff = np.abs(off_vals - np.median(off_vals))
    valid = diff < np.percentile(diff, 90)
    if np.sum(valid) < 5:
        valid = np.ones(len(matches), dtype=bool)

    slope, intercept = np.polyfit(t_vals[valid], off_vals[valid], 1)
    y_pred = slope * t_vals[valid] + intercept
    y_true = off_vals[valid]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-6))

    ppm = slope * 1e6
    if abs(slope - 0.001001) < 0.00015:
        diag = "23.976 vs 24.000 fps mismatch (Resample audio x1001/1000)"
    elif abs(slope - (-0.001000)) < 0.00015:
        diag = "24.000 vs 23.976 fps mismatch (Resample audio x1000/1001)"
    elif abs(slope - (-0.040000)) < 0.002:
        diag = "25.000 to 24.000 fps PAL conversion (Resample audio x24/25)"
    elif abs(slope - 0.041666) < 0.002:
        diag = "24.000 to 25.000 fps PAL conversion (Resample audio x25/24)"
    elif abs(ppm) < 25.0:
        diag = "1:1 True speed match (< 25 ppm drift)"
    else:
        diag = f"Uncalibrated clock drift: {ppm:+.1f} ppm ({slope:+.6f} s/s)"

    return {
        "slope": float(slope),
        "base_offset": float(intercept),
        "ppm": float(ppm),
        "r_squared": float(r2),
        "diagnosis": diag,
    }
