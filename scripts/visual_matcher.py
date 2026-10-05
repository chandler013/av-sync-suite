"""
Visual Scene Matcher & Sequence Cross-Correlator.
Performs language-independent visual frame matching across video masters.
"""

import os
import subprocess
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from .utils import FFMPEG


def extract_thumbnails(
    video_path: str,
    raw_out_path: str,
    fps_sample: float = 2.0,
    width: int = 64,
    height: int = 36,
    crop_factor: float = 0.8,
) -> int:
    """
    Extracts small grayscale video thumbnails at a coarse framerate (default 2fps)
    into a raw byte file for fast correlation.
    Returns total number of extracted frames.
    """
    os.makedirs(os.path.dirname(os.path.abspath(raw_out_path)), exist_ok=True)
    crop_vf = f"crop=iw*{crop_factor}:ih*{crop_factor}," if crop_factor < 1.0 else ""
    vf = f"fps={fps_sample},{crop_vf}scale={width}:{height},format=gray"

    cmd = [
        FFMPEG, "-y",
        "-i", video_path,
        "-vf", vf,
        "-f", "rawvideo",
        raw_out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    size_bytes = os.path.getsize(raw_out_path)
    frame_size = width * height
    return size_bytes // frame_size


def load_raw_frames(raw_path: str, width: int = 64, height: int = 36) -> np.ndarray:
    """Loads raw grayscale bytes into uint8 numpy array of shape (N, H, W)."""
    with open(raw_path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((-1, height, width))


def find_visual_matches(
    target_frames: np.ndarray,
    donor_frames: np.ndarray,
    fps_sample: float = 2.0,
    win_dur: float = 4.0,
    step_sec: float = 1.0,
    initial_offset: float = 0.0,
    min_score: float = 0.38,
    search_window_left: float = 20.0,
    search_window_right: float = 40.0,
) -> List[Dict[str, float]]:
    """
    Multi-frame SEQUENCE cross-correlation across video timelines.
    Uses exponential moving average to lock onto the expected offset trajectory.
    """
    win_len = int(round(win_dur * fps_sample))
    n_t = len(target_frames)
    n_d = len(donor_frames)

    if n_t < win_len or n_d < win_len:
        return []

    # Pre-normalize donor frames
    d_means = np.mean(donor_frames, axis=(1, 2), keepdims=True)
    d_stds = np.std(donor_frames, axis=(1, 2), keepdims=True) + 1e-4
    d_norm = (donor_frames - d_means) / d_stds

    checkpoints = np.arange(0.0, (n_t / fps_sample) - win_dur, step_sec)
    matches: List[Dict[str, float]] = []
    expected_off = initial_offset

    for t_sec in checkpoints:
        t_idx = int(round(t_sec * fps_sample))
        chunk_t = target_frames[t_idx : t_idx + win_len]

        # Skip low variance segments (black cards, fadeouts)
        if np.std(chunk_t) < 4.0:
            continue

        chunk_t_norm = (chunk_t - np.mean(chunk_t, axis=(1, 2), keepdims=True)) / (
            np.std(chunk_t, axis=(1, 2), keepdims=True) + 1e-4
        )

        d_start = max(0, int((t_sec + expected_off - search_window_left) * fps_sample))
        d_end = min(n_d - win_len, int((t_sec + expected_off + search_window_right) * fps_sample))
        if d_end <= d_start:
            continue

        best_score, best_k = -1.0, d_start
        for k in range(d_start, d_end):
            score = float(np.mean(d_norm[k : k + win_len] * chunk_t_norm))
            if score > best_score:
                best_score, best_k = score, k

        offset_sec = (best_k / fps_sample) - t_sec
        if best_score >= min_score:
            matches.append({"t": float(t_sec), "offset": float(offset_sec), "score": float(best_score)})
            expected_off = 0.90 * expected_off + 0.10 * offset_sec

    return matches


def build_scene_map(
    matches: List[Dict[str, float]],
    target_duration: float,
    win_dur: float = 4.0,
    max_gap: float = 12.0,
    max_offset_diff: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    Clusters dense matches into continuous scene blocks and bridges timeline gaps
    to produce a continuous, gapless [0.0, target_duration] scene map.
    """
    if not matches:
        return [{"start": 0.0, "end": target_duration, "offset": 0.0, "score": 0.0, "is_gap": True}]

    raw_scenes: List[Dict[str, Any]] = []
    curr = [matches[0]]

    for m in matches[1:]:
        if (
            abs(m["offset"] - curr[-1]["offset"]) <= max_offset_diff
            and (m["t"] - curr[-1]["t"]) <= max_gap
        ):
            curr.append(m)
        else:
            if len(curr) >= 2:
                t0 = curr[0]["t"]
                t1 = curr[-1]["t"] + win_dur
                med_off = float(np.median([x["offset"] for x in curr]))
                avg_score = float(np.mean([x["score"] for x in curr]))
                raw_scenes.append({"start": t0, "end": t1, "offset": med_off, "score": avg_score, "is_gap": False})
            curr = [m]

    if len(curr) >= 2:
        t0 = curr[0]["t"]
        t1 = curr[-1]["t"] + win_dur
        med_off = float(np.median([x["offset"] for x in curr]))
        avg_score = float(np.mean([x["score"] for x in curr]))
        raw_scenes.append({"start": t0, "end": t1, "offset": med_off, "score": avg_score, "is_gap": False})

    if not raw_scenes:
        # Fallback to median offset of all matches
        med_off = float(np.median([x["offset"] for x in matches]))
        return [{"start": 0.0, "end": target_duration, "offset": med_off, "score": 0.5, "is_gap": False}]

    # Bridge gaps and ensure timeline covers 0.0 to target_duration
    filled_scenes: List[Dict[str, Any]] = []
    cursor = 0.0

    for sc in raw_scenes:
        s0, s1 = sc["start"], sc["end"]
        if s0 > cursor:
            # Fill gap before current scene
            fill_offset = sc["offset"]
            filled_scenes.append({
                "start": cursor,
                "end": s0,
                "offset": fill_offset,
                "score": sc["score"],
                "is_gap": True,
            })
        filled_scenes.append(sc)
        cursor = max(cursor, s1)

    if cursor < target_duration:
        filled_scenes.append({
            "start": cursor,
            "end": target_duration,
            "offset": filled_scenes[-1]["offset"] if filled_scenes else 0.0,
            "score": filled_scenes[-1]["score"] if filled_scenes else 0.0,
            "is_gap": True,
        })

    # Merge adjacent scenes with identical offsets
    merged: List[Dict[str, Any]] = []
    for sc in filled_scenes:
        if merged and abs(merged[-1]["offset"] - sc["offset"]) < 0.01:
            merged[-1]["end"] = sc["end"]
        else:
            merged.append(sc)

    return merged
