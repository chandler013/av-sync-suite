"""
Utility functions and external binary discovery for av-sync-suite.
"""

import os
import sys
import shutil
import json
import subprocess
from typing import Dict, Any, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_binary(name: str) -> str:
    """Finds binary path across PATH and common installation directories."""
    found = shutil.which(name)
    if found:
        return found

    windows_fallbacks = {
        "mkvmerge": [
            r"C:\Program Files\MKVToolNix\mkvmerge.exe",
            r"C:\Program Files (x86)\MKVToolNix\mkvmerge.exe",
        ],
        "mkvpropedit": [
            r"C:\Program Files\MKVToolNix\mkvpropedit.exe",
            r"C:\Program Files (x86)\MKVToolNix\mkvpropedit.exe",
        ],
        "ffmpeg": [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
        ],
        "ffprobe": [
            r"C:\ffmpeg\bin\ffprobe.exe",
            os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"),
        ],
    }

    base = os.path.splitext(name.lower())[0]
    if base in windows_fallbacks:
        for path in windows_fallbacks[base]:
            if os.path.isfile(path):
                return path

    return name


FFMPEG = find_binary("ffmpeg")
FFPROBE = find_binary("ffprobe")
MKVMERGE = find_binary("mkvmerge")
MKVPROPEDIT = find_binary("mkvpropedit")


def probe_file(file_path: str) -> Dict[str, Any]:
    """Probes media file streams and container format using ffprobe."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(res.stdout)


def get_stream_info(probe_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts high-level summary of video and audio tracks."""
    info: Dict[str, Any] = {
        "duration": float(probe_data.get("format", {}).get("duration", 0.0)),
        "video": None,
        "audio": [],
    }

    for stream in probe_data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video" and not info["video"]:
            fps_str = stream.get("r_frame_rate", "0/1")
            fps = 0.0
            if "/" in fps_str:
                num, den = fps_str.split("/")
                den = float(den)
                fps = float(num) / den if den != 0 else 0.0
            else:
                fps = float(fps_str or 0.0)

            info["video"] = {
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": fps,
                "fps_str": fps_str,
            }
        elif codec_type == "audio":
            tags = stream.get("tags", {})
            info["audio"].append({
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "channels": stream.get("channels"),
                "channel_layout": stream.get("channel_layout", f"{stream.get('channels')}ch"),
                "sample_rate": int(stream.get("sample_rate", 48000)),
                "language": tags.get("language", "und"),
                "title": tags.get("title", ""),
            })

    return info


def format_seconds(seconds: float) -> str:
    """Formats float seconds into HH:MM:SS.mmm format."""
    sign = "-" if seconds < 0 else ""
    sec = abs(seconds)
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    remainder = sec % 60
    return f"{sign}{hours:02d}:{minutes:02d}:{remainder:06.3f}"
