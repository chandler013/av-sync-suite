"""
Native Framerate Sub-Frame Calibrator.
Snaps coarse 2fps offsets to exact 0-frame peak correlation (< 40ms error).
"""

import subprocess
from typing import Tuple, List, Dict, Any
import numpy as np
from .utils import FFMPEG


def calibrate_scene_offset_native_fps(
    target_video: str,
    donor_video: str,
    t_target: float,
    nominal_offset: float,
    fps: float = 24.0,
    search_frames: int = 24,
    crop_factor: float = 0.8,
) -> Tuple[float, int, float]:
    """
    Extracts frames at native framerate and finds peak cross-correlation shift.
    Returns:
        (refined_offset_sec, best_shift_frames, peak_score)
    """
    t_donor = t_target + nominal_offset
    win_len = int(fps * 0.8)  # ~0.8s sequence
    crop_vf = f"crop=iw*{crop_factor}:ih*{crop_factor}," if crop_factor < 1.0 else ""

    cmd_tgt = [
        FFMPEG, "-y",
        "-ss", str(t_target),
        "-i", target_video,
        "-frames:v", str(int(fps)),
        "-vf", f"{crop_vf}scale=96:48,format=gray",
        "-f", "rawvideo", "-",
    ]

    cmd_don = [
        FFMPEG, "-y",
        "-ss", str(max(0.0, t_donor - 1.0)),
        "-i", donor_video,
        "-frames:v", str(int(fps * 3)),
        "-vf", f"{crop_vf}scale=96:48,format=gray",
        "-f", "rawvideo", "-",
    ]

    p1 = subprocess.Popen(cmd_tgt, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(cmd_don, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_tgt, _ = p1.communicate()
    raw_don, _ = p2.communicate()

    if not raw_tgt or not raw_don:
        return nominal_offset, 0, 0.0

    frame_pixels = 96 * 48
    f_tgt = np.frombuffer(raw_tgt, dtype=np.uint8).reshape((-1, frame_pixels)).astype(np.float32)
    f_don = np.frombuffer(raw_don, dtype=np.uint8).reshape((-1, frame_pixels)).astype(np.float32)

    if len(f_tgt) < win_len or len(f_don) < win_len:
        return nominal_offset, 0, 0.0

    f_tgt_norm = (f_tgt - f_tgt.mean(axis=1, keepdims=True)) / (f_tgt.std(axis=1, keepdims=True) + 1e-4)
    f_don_norm = (f_don - f_don.mean(axis=1, keepdims=True)) / (f_don.std(axis=1, keepdims=True) + 1e-4)

    nominal_idx = int(fps)  # Because donor extracted from t_donor - 1.0s
    best_score = -1.0
    best_shift = 0

    for shift in range(-search_frames, search_frames + 1):
        idx_don = nominal_idx + shift
        if 0 <= idx_don and idx_don + win_len <= len(f_don_norm):
            score = float(np.mean(f_tgt_norm[:win_len] * f_don_norm[idx_don : idx_don + win_len]))
            if score > best_score:
                best_score = score
                best_shift = shift

    refined_offset = nominal_offset + (best_shift / fps)
    return float(refined_offset), int(best_shift), float(best_score)


def refine_scene_map(
    scene_map: List[Dict[str, Any]],
    target_video: str,
    donor_video: str,
    fps: float = 24.0,
    min_confidence_score: float = 0.40,
) -> List[Dict[str, Any]]:
    """
    Iterates over all scenes in scene_map, samples midpoints, and refines the offsets
    to exact 0-frame peak precision at native framerate.
    """
    refined_map: List[Dict[str, Any]] = []

    for sc in scene_map:
        if sc.get("is_gap", False) and sc.get("score", 0.0) < 0.2:
            refined_map.append(sc)
            continue

        s0, s1 = sc["start"], sc["end"]
        mid_t = (s0 + s1) / 2.0
        nom_offset = sc["offset"]

        try:
            ref_offset, shift, score = calibrate_scene_offset_native_fps(
                target_video, donor_video, mid_t, nom_offset, fps=fps
            )
            if score >= min_confidence_score:
                sc_refined = dict(sc)
                sc_refined["offset"] = ref_offset
                sc_refined["shift_frames"] = shift
                sc_refined["refined_score"] = score
                refined_map.append(sc_refined)
            else:
                refined_map.append(sc)
        except Exception:
            refined_map.append(sc)

    return refined_map
