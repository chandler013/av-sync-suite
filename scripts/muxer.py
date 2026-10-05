"""
MKV Remuxer & Metadata Sanitizer.
Muxes untouched video stream with synchronized audio tracks and sanitizes container tags.
"""

import os
import subprocess
from typing import List, Dict, Any, Optional
from .utils import MKVMERGE, MKVPROPEDIT, FFMPEG


def encode_ac3(input_audio: str, output_ac3: str, bitrate: str = "448k") -> None:
    """Encodes lossless multi-channel FLAC/WAV to AC-3 preserving channel layout."""
    os.makedirs(os.path.dirname(os.path.abspath(output_ac3)), exist_ok=True)
    cmd = [
        FFMPEG, "-y",
        "-i", input_audio,
        "-c:a", "ac3",
        "-b:a", bitrate,
        output_ac3,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def mux_final_mkv(
    target_video: str,
    synced_audio: str,
    output_mkv: str,
    movie_title: str,
    audio_track_name: str = "Telugu [DD 5.1 @ 448 kbps - Synced]",
    audio_language: str = "tel",
    keep_target_audio: bool = True,
    target_audio_name: str = "Original Audio",
    target_audio_language: str = "hin",
) -> None:
    """
    Losslessly muxes the untouched target video with the synchronized audio track.
    Applies clean ISO 639-2 language tags and sanitizes container metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_mkv)), exist_ok=True)

    cmd = [
        MKVMERGE,
        "-o", output_mkv,
        "--title", movie_title,
        # Video stream from target
        "--video-tracks", "0",
        "--no-audio",
        "--no-subtitles",
        target_video,
        # Synced audio
        "--track-name", f"0:{audio_track_name}",
        "--language", f"0:{audio_language}",
        "--default-track", "0:yes",
        synced_audio,
    ]

    if keep_target_audio:
        cmd.extend([
            "--no-video",
            "--no-subtitles",
            "--audio-tracks", "0",
            "--track-name", f"0:{target_audio_name}",
            "--language", f"0:{target_audio_language}",
            "--default-track", "0:no",
            target_video,
        ])

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Sanitize container tags
    if os.path.isfile(MKVPROPEDIT):
        sanitize_cmd = [
            MKVPROPEDIT,
            output_mkv,
            "--tags", "all:",
            "--edit", "info",
            "--set", f"title={movie_title}",
        ]
        subprocess.run(sanitize_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
