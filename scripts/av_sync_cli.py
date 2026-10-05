"""
Unified Command-Line Interface for av-sync-suite.
Supports standalone execution or AI Agent invocation.
"""

import os
import sys
import argparse
import json
import subprocess
from typing import Optional

# Allow relative or direct module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import probe_file, get_stream_info, format_seconds, FFMPEG, MKVMERGE
from scripts.visual_matcher import extract_thumbnails, load_raw_frames, find_visual_matches, build_scene_map
from scripts.native_calibrator import refine_scene_map
from scripts.acoustic_correlator import extract_reference_audio, find_acoustic_matches, analyze_clock_drift
from scripts.stream_assembler import slice_and_stream_audio
from scripts.muxer import encode_ac3, mux_final_mkv


def cmd_probe(args: argparse.Namespace) -> None:
    print(f"--- Probing Target: {args.target} ---")
    tgt_info = get_stream_info(probe_file(args.target))
    print(f"Duration: {format_seconds(tgt_info['duration'])} ({tgt_info['duration']:.2f}s)")
    if tgt_info["video"]:
        v = tgt_info["video"]
        print(f"Video: {v['width']}x{v['height']} @ {v['fps']:.3f} fps ({v['codec']})")
    for a in tgt_info["audio"]:
        print(f"Audio #{a['index']}: {a['codec']} {a['channel_layout']} @ {a['sample_rate']}Hz [{a['language']}] {a['title']}")

    if args.donor:
        print(f"\n--- Probing Donor: {args.donor} ---")
        don_info = get_stream_info(probe_file(args.donor))
        print(f"Duration: {format_seconds(don_info['duration'])} ({don_info['duration']:.2f}s)")
        if don_info["video"]:
            v = don_info["video"]
            print(f"Video: {v['width']}x{v['height']} @ {v['fps']:.3f} fps ({v['codec']})")
        for a in don_info["audio"]:
            print(f"Audio #{a['index']}: {a['codec']} {a['channel_layout']} @ {a['sample_rate']}Hz [{a['language']}] {a['title']}")


def cmd_visual_map(args: argparse.Namespace) -> None:
    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    raw_tgt = os.path.join(tmp_dir, "tgt_2fps.raw")
    raw_don = os.path.join(tmp_dir, "don_2fps.raw")

    print("[1/4] Extracting 2fps thumbnails from target video...")
    n_tgt = extract_thumbnails(args.target, raw_tgt, fps_sample=args.fps_sample, crop_factor=args.crop_factor)
    print(f"      Extracted {n_tgt} target frames.")

    print("[2/4] Extracting 2fps thumbnails from donor video...")
    n_don = extract_thumbnails(args.donor, raw_don, fps_sample=args.fps_sample, crop_factor=args.crop_factor)
    print(f"      Extracted {n_don} donor frames.")

    print("[3/4] Running multi-frame sequence cross-correlation...")
    frames_tgt = load_raw_frames(raw_tgt)
    frames_don = load_raw_frames(raw_don)

    matches = find_visual_matches(
        frames_tgt,
        frames_don,
        fps_sample=args.fps_sample,
        win_dur=args.win_dur,
        initial_offset=args.initial_offset,
        min_score=args.min_score,
    )
    print(f"      Found {len(matches)} valid sequence matches.")

    print("[4/4] Building gapless discrete scene map...")
    tgt_info = get_stream_info(probe_file(args.target))
    scene_map = build_scene_map(matches, target_duration=tgt_info["duration"])

    with open(args.out_map, "w", encoding="utf-8") as f:
        json.dump(scene_map, f, indent=2)

    print(f"SUCCESS: Scene map saved with {len(scene_map)} scenes -> {args.out_map}")


def cmd_refine_map(args: argparse.Namespace) -> None:
    with open(args.in_map, "r", encoding="utf-8") as f:
        scene_map = json.load(f)

    tgt_info = get_stream_info(probe_file(args.target))
    fps = tgt_info["video"]["fps"] if tgt_info["video"] else 24.0

    print(f"Refining {len(scene_map)} scenes at native {fps:.3f} fps precision...")
    refined = refine_scene_map(scene_map, args.target, args.donor, fps=fps)

    with open(args.out_map, "w", encoding="utf-8") as f:
        json.dump(refined, f, indent=2)

    print(f"SUCCESS: Refined scene map saved -> {args.out_map}")


def cmd_acoustic_map(args: argparse.Namespace) -> None:
    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    ref_tgt = os.path.join(tmp_dir, "ref_tgt_16k.wav")
    ref_don = os.path.join(tmp_dir, "ref_don_16k.wav")

    print("[1/3] Extracting 16kHz Center-channel reference tracks...")
    extract_reference_audio(args.target, ref_tgt, stream_index=args.target_audio_index)
    extract_reference_audio(args.donor, ref_don, stream_index=args.donor_audio_index)

    print("[2/3] Correlating 100Hz RMS energy envelopes...")
    matches = find_acoustic_matches(ref_tgt, ref_don)
    print(f"      Found {len(matches)} acoustic match points.")

    print("[3/3] Analyzing clock drift & framerate ratio...")
    drift = analyze_clock_drift(matches)
    print(f"      Drift Diagnosis: {drift['diagnosis']}")
    print(f"      Linear Slope: {drift['slope']:+.7f} s/s ({drift['ppm']:+.1f} ppm)")
    print(f"      Base Offset:  {drift['base_offset']:+.3f}s (R2={drift['r_squared']:.4f})")

    if args.out_map:
        with open(args.out_map, "w", encoding="utf-8") as f:
            json.dump({"matches": matches, "drift_analysis": drift}, f, indent=2)
        print(f"Acoustic map saved -> {args.out_map}")


def cmd_assemble(args: argparse.Namespace) -> None:
    with open(args.scene_map, "r", encoding="utf-8") as f:
        scene_map = json.load(f)

    print(f"Streaming donor audio assembly from {args.donor_audio}...")
    res = slice_and_stream_audio(
        donor_path=args.donor_audio,
        out_flac_path=args.output,
        scene_map=scene_map,
        target_duration_sec=args.target_duration,
        sample_rate=args.sample_rate,
        target_fallback_path=args.fallback_audio,
    )
    print(f"SUCCESS: Audio assembled -> {res['output_path']}")
    print(f"Total Written: {res['total_written_samples']} samples | Drift: {res['drift_ms']:.2f} ms")


def cmd_mux(args: argparse.Namespace) -> None:
    synced_audio = args.audio
    if args.encode_ac3:
        ac3_out = os.path.splitext(synced_audio)[0] + ".ac3"
        print(f"Encoding lossless audio to AC-3 ({args.ac3_bitrate})...")
        encode_ac3(synced_audio, ac3_out, bitrate=args.ac3_bitrate)
        synced_audio = ac3_out

    print(f"Muxing final MKV -> {args.output}")
    mux_final_mkv(
        target_video=args.target,
        synced_audio=synced_audio,
        output_mkv=args.output,
        movie_title=args.title,
        audio_track_name=args.track_name,
        audio_language=args.language,
        keep_target_audio=args.keep_target_audio,
        target_audio_name=args.target_track_name,
        target_audio_language=args.target_language,
    )
    print(f"SUCCESS: Final container authored -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="av-sync-suite: Audio-Video Synchronization & Remuxing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # probe
    p_probe = subparsers.add_parser("probe", help="Probe target and donor video/audio tracks")
    p_probe.add_argument("target", help="Path to target video file")
    p_probe.add_argument("donor", nargs="?", default=None, help="Path to donor video file")
    p_probe.set_defaults(func=cmd_probe)

    # visual-map
    p_vmap = subparsers.add_parser("visual-map", help="Compute visual sequence scene map")
    p_vmap.add_argument("--target", required=True, help="Target video (reference picture)")
    p_vmap.add_argument("--donor", required=True, help="Donor video (source of audio)")
    p_vmap.add_argument("--out-map", required=True, help="Output scene_map.json path")
    p_vmap.add_argument("--tmp-dir", default="tmp", help="Scratch directory for thumbnails")
    p_vmap.add_argument("--fps-sample", type=float, default=2.0, help="Thumbnail sample fps (default: 2.0)")
    p_vmap.add_argument("--win-dur", type=float, default=4.0, help="Sequence window duration in sec (default: 4.0)")
    p_vmap.add_argument("--initial-offset", type=float, default=0.0, help="Estimated initial offset in sec")
    p_vmap.add_argument("--crop-factor", type=float, default=0.8, help="Inner crop factor (default: 0.8)")
    p_vmap.add_argument("--min-score", type=float, default=0.38, help="Minimum sequence correlation score")
    p_vmap.set_defaults(func=cmd_visual_map)

    # refine-map
    p_refine = subparsers.add_parser("refine-map", help="Refine scene map offsets at native framerate")
    p_refine.add_argument("--target", required=True, help="Target video")
    p_refine.add_argument("--donor", required=True, help="Donor video")
    p_refine.add_argument("--in-map", required=True, help="Input coarse scene_map.json")
    p_refine.add_argument("--out-map", required=True, help="Output refined scene_map.json")
    p_refine.set_defaults(func=cmd_refine_map)

    # acoustic-map
    p_ac = subparsers.add_parser("acoustic-map", help="Compute acoustic correlation map for same-language tracks")
    p_ac.add_argument("--target", required=True, help="Target video")
    p_ac.add_argument("--donor", required=True, help="Donor video")
    p_ac.add_argument("--target-audio-index", type=int, default=0, help="Target audio track index")
    p_ac.add_argument("--donor-audio-index", type=int, default=0, help="Donor audio track index")
    p_ac.add_argument("--tmp-dir", default="tmp", help="Scratch directory")
    p_ac.add_argument("--out-map", default=None, help="Optional output JSON map")
    p_ac.set_defaults(func=cmd_acoustic_map)

    # assemble
    p_as = subparsers.add_parser("assemble", help="Assemble multi-channel audio directly to disk")
    p_as.add_argument("--donor-audio", required=True, help="Extracted donor FLAC/WAV file")
    p_as.add_argument("--scene-map", required=True, help="Path to scene_map.json")
    p_as.add_argument("--target-duration", type=float, required=True, help="Target video duration in seconds")
    p_as.add_argument("--output", required=True, help="Output synced FLAC file")
    p_as.add_argument("--sample-rate", type=int, default=48000, help="Sample rate (default: 48000)")
    p_as.add_argument("--fallback-audio", default=None, help="Target reference audio for hybrid fill")
    p_as.set_defaults(func=cmd_assemble)

    # mux
    p_mux = subparsers.add_parser("mux", help="Mux target video and synced audio into final MKV")
    p_mux.add_argument("--target", required=True, help="Target video file")
    p_mux.add_argument("--audio", required=True, help="Synced audio file (FLAC or AC3)")
    p_mux.add_argument("--output", required=True, help="Final output MKV file")
    p_mux.add_argument("--title", required=True, help="Clean movie title")
    p_mux.add_argument("--track-name", default="Telugu [DD 5.1 @ 448 kbps - Synced]", help="Track name")
    p_mux.add_argument("--language", default="tel", help="ISO 639-2 language code (e.g. tel, hin, eng)")
    p_mux.add_argument("--encode-ac3", action="store_true", help="Encode to AC-3 before muxing")
    p_mux.add_argument("--ac3-bitrate", default="448k", help="AC-3 bitrate (default: 448k)")
    p_mux.add_argument("--keep-target-audio", action="store_true", default=True, help="Retain original target audio")
    p_mux.add_argument("--target-track-name", default="Original Audio", help="Target audio track name")
    p_mux.add_argument("--target-language", default="hin", help="Target audio language code")
    p_mux.set_defaults(func=cmd_mux)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
