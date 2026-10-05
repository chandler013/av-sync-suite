# 🎬 AV Sync Suite (`av-sync-suite`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Supported-green.svg)](https://ffmpeg.org/)
[![MKVToolNix](https://img.shields.io/badge/MKVToolNix-Supported-purple.svg)](https://mkvtoolnix.download/)
[![Agent Skill](https://img.shields.io/badge/AI%20Agent-Skill%20Ready-brightgreen.svg)](SKILL.md)

Production-grade toolkit and AI agent skill for synchronizing audio tracks across different video masters (e.g. physical DVD/Blu-ray 5.1 surround onto 1080p/4K WEB-DL masters, regional dub audio alignment, or extended director's cuts).

Equally usable as a **turnkey CLI application** and as an **AI Coding Agent Skill** (Antigravity, Claude Code, Cursor, Codex, OpenCode).

---

## ⚡ Key Highlights & Core Capabilities

- **Language-Independent Visual Frame Mapping**: Correlates multi-frame video sequences rather than acoustic waveforms. Perfect for syncing dubbed audio or multi-language releases where speech content is completely different.
- **Native-Framerate Sub-Frame Peak Refinement**: Snaps coarse 2fps sampling offsets to exact **0-frame peak precision (< 40ms error)**, eliminating lip-sync jitter.
- **Zero-Memory Streaming Audio Assembly**: Slices multi-channel audio directly to disk via block streaming with **in-place 8–20ms cosine crossfades**, consuming **< 50MB RAM** and guaranteeing **zero cumulative sample loss** across 2.5-hour films.
- **Hybrid Stream Handling**: Detects target-exclusive scenes (uncut streaming comedy tracks, extended dialogues) and seamlessly fills them with reference audio upmixed to 5.1.
- **Center-Channel Isolated Acoustic Correlation**: For same-language tracks, extracts the isolated Center Channel (Channel 2 / FC) to prevent surround phase cancellation, computing 100Hz RMS energy envelopes in seconds.
- **Clock Drift & Framerate Ratio Detection**: Linear regression on anchor points detects 23.976 vs 24.000 fps ($1000/1001$) pull-up/pulldown, PAL 25.000 fps speedup/slowdown, and uncalibrated clock drift.
- **Lossless Keyframe Splicing & Ad Removal**: Removes broadcaster ad-pods or distributor intro logos bit-for-bit with `mkvmerge --split parts:...` with zero re-encoding.

---

## 📁 Repository Structure

```text
av-sync-suite/
├── SKILL.md                          # Primary AI agent skill specification
├── skills/
│   ├── av-sync-suite/
│   │   └── SKILL.md                  # Standard skill directory layout
│   └── sync-different-runtimes/
│       └── SKILL.md                  # Advanced companion skill for varying runtimes
├── scripts/
│   ├── __init__.py
│   ├── av_sync_cli.py                # Master CLI interface
│   ├── utils.py                      # Binary discovery (ffmpeg, mkvmerge) & media probe
│   ├── visual_matcher.py             # 2fps thumbnail extraction & sequence correlation
│   ├── native_calibrator.py          # Native-framerate sub-frame peak refinement
│   ├── acoustic_correlator.py        # Center-channel RMS envelope correlation & drift fit
│   ├── stream_assembler.py           # Zero-memory streaming multi-channel FLAC slicer
│   └── muxer.py                      # MKV remuxing & metadata sanitizer
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ Requirements & Installation

### External Binaries
Make sure the following tools are installed and present in your system `PATH` (or default Windows installation paths):
- **FFmpeg / FFprobe** (`ffmpeg`, `ffprobe`)
- **MKVToolNix** (`mkvmerge`, `mkvpropedit`)

### Python Dependencies
```bash
git clone https://github.com/chandler013/av-sync-suite.git
cd av-sync-suite
pip install -r requirements.txt
```

---

## 🚀 Quick Start (CLI Usage)

### 1. Probe Source Files
Examine durations, video resolutions, frame rates, and audio channel layouts:
```bash
python scripts/av_sync_cli.py probe target.mkv donor.mkv
```

### 2. Generate Visual Scene Map (Cross-Master / Dubbed Sources)
Extract 2fps thumbnails and compute continuous sequence offsets:
```bash
python scripts/av_sync_cli.py visual-map \
  --target target.mkv \
  --donor donor.mkv \
  --out-map tmp/scene_map.json \
  --initial-offset 0.0
```

### 3. Refine Scene Offsets to 0-Frame Precision (< 40ms error)
Refine coarse offsets at the target video's native framerate (24.0 or 25.0 fps):
```bash
python scripts/av_sync_cli.py refine-map \
  --target target.mkv \
  --donor donor.mkv \
  --in-map tmp/scene_map.json \
  --out-map tmp/scene_map_refined.json
```

### 4. Assemble Multi-Channel Audio
Stream the donor audio through the refined scene map directly to 24-bit FLAC:
```bash
python scripts/av_sync_cli.py assemble \
  --donor-audio tmp/donor_5ch_48k.flac \
  --scene-map tmp/scene_map_refined.json \
  --target-duration 8420.50 \
  --output tmp/synced_audio_5ch.flac
```

### 5. Encode & Remux to Final MKV
Encode synced audio to AC-3 (or keep FLAC) and losslessly mux with target video:
```bash
python scripts/av_sync_cli.py mux \
  --target target.mkv \
  --audio tmp/synced_audio_5ch.flac \
  --output "output/Movie (Year) - 1080p AVC DD5.1 [Synced Audio].mkv" \
  --title "Movie (Year)" \
  --track-name "Telugu [DD 5.1 @ 448 kbps - Synced]" \
  --language tel \
  --encode-ac3 \
  --ac3-bitrate 448k
```

---

## 🤖 Using as an AI Agent Skill

This repository adheres to the **Antigravity / Agentic Skill Standard**. You can integrate this skill into any AI coding environment:

### Antigravity / Google Stitch
Copy or symlink `skills/av-sync-suite` and `skills/sync-different-runtimes` into your global or project customization root:
```bash
# Global
cp -r skills/av-sync-suite ~/.gemini/config/skills/
cp -r skills/sync-different-runtimes ~/.gemini/config/skills/

# Or per-workspace
cp -r skills/av-sync-suite .agents/skills/
```

### Claude Code / Cursor / Codex
Point your AI agent to `SKILL.md`. The agent will autonomously:
1. Probe container streams and frame rates.
2. Select whether to use Visual Scene Mapping or Acoustic Correlation.
3. Handle framerate conversion ($25 \leftrightarrow 24$, $23.976 \leftrightarrow 24.000$).
4. Slice and stream multi-channel audio without memory leaks.
5. Audit and verify sync across multiple timeline checkpoints before delivering.

---

## 🛡️ The Golden Rules of Audio-Video Sync

1. **The Cardinal Rule: Never Split Target Video.**  
   Always keep the target video stream 100% byte-identical to source. Perform all cuts, trims, and offsets strictly on audio streams. Splitting video causes frame-boundary jitter, millisecond timestamp drift, and visual stutter.
2. **Never Average 5.1 Audio to Mono.**  
   Film dialogue is isolated in the **Center Channel (Channel 2)**. Averaging surround and LFE channels washes out speech transients and causes destructive phase cancellation. Always extract Center channel (`pan=mono|c0=FC`) for correlation.
3. **Never Accumulate Arrays in Memory.**  
   Concatenating multi-gigabyte numpy arrays in loops causes RAM explosion and freezes the system. Always use streaming block writers.
4. **Never Use Continuous Sample Interpolation.**  
   Resampling audio across continuous spline curves introduces Doppler flutter and metallic comb filtering. Use discrete integer-sample slicing with boundary micro-crossfades.

---

## 📜 License

Released under the [MIT License](LICENSE).
