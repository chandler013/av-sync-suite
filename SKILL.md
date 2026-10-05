---
name: av-sync-suite
description: >-
  Audio-video processing toolkit for synchronizing audio tracks across different
  video masters (DVD, Blu-ray, WEB-DL, dubbed sources). Covers DVD remuxing,
  multi-channel audio sync via visual scene mapping, subtitle alignment, video
  deinterlacing, framerate conversion, and clean media tagging.
---

# Audio-Video Synchronization & Processing Skill (`av-sync-suite`)

Synchronize audio tracks between different video masters (e.g. DVD 5.0 Telugu audio onto a
1080p WEB-DL Hindi video, or a dubbed MP4's Hindi audio onto a Telugu Blu-ray). Also covers
DVD remuxing, deinterlacing, framerate conversion, and metadata hygiene.

**This skill does NOT do studio remastering.** It synchronizes, remuxes, and labels. Name
output files accordingly — no "remaster" or "master" language.

---

## 0. Working Directory Convention

```text
project_root/
├── source_video_a.mkv              # never modified in place
├── source_video_b.mkv              # never modified in place
├── output/                         # final deliverables only
│   └── Title (Year) - 1080p AVC DD5.0 [Synced Audio].mkv
└── tmp/                            # everything else — safe to delete after a run
    ├── ref_target_16k.wav          # 16kHz mono reference from target video
    ├── ref_donor_16k.wav           # 16kHz mono reference from donor video
    ├── target_vid_2fps.raw         # grayscale thumbnails for visual matching
    ├── donor_vid_2fps.raw
    ├── scene_map.json              # computed visual scene offset map
    ├── donor_5ch_48k.flac          # extracted multi-channel donor audio
    ├── synced_audio_5ch.flac       # assembled synced audio
    ├── synced_audio_5ch.ac3        # encoded final audio
    └── *.py                        # scratch scripts (deleted after use)
```

Rules:
1. **Never write scratch files into `project_root/` directly** — always under `tmp/`.
2. **Never modify source files.** The source videos stay 100% untouched.
3. Only move a finished file into `output/` after it passes the Section 7 checklist.
4. Create `tmp/` and `output/` at the start of a run.
5. Delete scratch `.py` scripts from `tmp/` after they run successfully.

---

## 1. The Cardinal Rule: Never Split Video

**Never split or cut the target video.** Keep the target video stream byte-identical to
the source. Perform all trims, splices, and offsets strictly on the audio track(s).

Splitting video causes:
- Millisecond timestamp drift at rejoin points
- Frame-boundary mismatches
- Stutter and visual glitches at split points

---

## 2. Audio-to-Video Synchronization

This is the core workflow. There are two fundamentally different source relationships:

### Case A: Same-Content Sources (same movie, different masters)

The donor audio and target video contain the **same movie** but from different masters
(e.g. DVD vs WEB-DL, or a dubbed YouTube upload vs a streaming master). The masters
differ in:
- Intro/outro logos and credits
- Song sequences (extended or trimmed)
- Censorship cuts
- Scene ordering

**This is the common case.** Use the Visual Scene Mapping pipeline (Section 2.1).

### Case B: Same-Content, Same-Speed, Fixed Offset

Both sources are from the same master with only a constant time offset (e.g. same
broadcast rip with different lead-in). A single cross-correlation pass gives one fixed
offset. Apply it with `mkvmerge --sync`.

---

### 2.1 Visual Scene Mapping Pipeline (Method A — The Reliable Method)

> **Why visual, not acoustic?**
>
> When syncing audio between different dubs (Hindi audio → Telugu video, or DVD Telugu
> audio → WEB-DL Hindi video), the dialogue content is **completely different**. Acoustic
> cross-correlation between different languages produces noisy, unreliable offset maps
> because speech formants, timing, and prosody differ between dubs. Background music and
> SFX overlap, but their correlation scores are weak and ambiguous.
>
> Visual frame matching is **language-independent** and produces unambiguous, high-confidence
> matches because the video frames are identical (same movie, different masters).

#### Step 1: Probe Both Sources

```python
# Always probe first. Never assume frame rates, durations, or channel layouts.
ffprobe -v quiet -print_format json -show_streams -show_format "target.mkv"
ffprobe -v quiet -print_format json -show_streams -show_format "donor.mkv"
```

Record:
- Frame rate (must be identical or known ratio — e.g. both 25.000 fps)
- Duration of each source
- Audio channel layout and codec of the donor track
- Sample rate

#### Step 2: Extract Grayscale Video Thumbnails

Extract tiny grayscale frames from both videos at 2 fps for fast visual correlation:

```python
w, h = 64, 36  # tiny thumbnails — fast to correlate
fps_sample = 2  # 2 frames per second

# Target video (e.g. 1080p WEB-DL)
ffmpeg -y -i "target.mkv" \
  -vf f"fps={fps_sample},crop=iw*0.8:ih*0.8,scale={w}:{h},format=gray" \
  -f rawvideo "tmp/target_vid_2fps.raw"

# Donor video (e.g. 576p DVD)
ffmpeg -y -i "donor.mkv" \
  -vf f"fps={fps_sample},crop=iw*0.8:ih*0.8,scale={w}:{h},format=gray" \
  -f rawvideo "tmp/donor_vid_2fps.raw"
```

The `crop=iw*0.8:ih*0.8` trims outer borders (letterbox bars, edge noise) so they don't
pollute the match.

> [!WARNING]
> If one source is Scope 2.38:1 inside a 16:9 container (black bars ~275px) and the other is cropped 2.08:1/16:9,
> `crop=iw*0.8:ih*0.8` leaves black bars on one and crops into actors on the other, dropping correlation to noise (< 0.28).
> For sources with different runtimes, micro-cuts, or aspect ratio mismatches, use the dedicated **`sync-different-runtimes`** skill.

#### Step 3: Dense Visual Sequence Correlation

For each 1-second checkpoint along the target video timeline, find the best-matching
position in the donor video by correlating a short video sequence (4–6 seconds of frames):

```python
import numpy as np

def find_visual_matches(target_frames, donor_frames, fps_sample=2,
                        win_dur=4.0, step_sec=1.0, initial_offset=0.0):
    """
    Returns list of {t, offset, score} dicts.

    CRITICAL DESIGN DECISIONS:
    - Use multi-frame SEQUENCE correlation (4-6 seconds), not single-frame matching.
      Single frames produce false matches on visually similar but temporally wrong scenes.
    - Track expected offset with exponential moving average so the search window stays
      narrow and doesn't wander into false matches.
    - Use normalized cross-correlation per frame, then average across the sequence.
    """
    win_len = int(win_dur * fps_sample)
    n_t = len(target_frames)
    n_d = len(donor_frames)

    # Precompute normalized donor frames
    d_means = np.mean(donor_frames, axis=(1, 2), keepdims=True)
    d_stds = np.std(donor_frames, axis=(1, 2), keepdims=True) + 1e-4
    d_norm = (donor_frames - d_means) / d_stds

    checkpoints = np.arange(0.0, (n_t / fps_sample) - win_dur, step_sec)
    matches = []
    expected_off = initial_offset

    for t_sec in checkpoints:
        t_idx = int(t_sec * fps_sample)
        chunk_t = target_frames[t_idx : t_idx + win_len]

        # Skip low-variance chunks (black frames, static cards)
        if np.std(chunk_t) < 4.0:
            continue

        # Normalize target chunk per-frame
        chunk_t_norm = (chunk_t - np.mean(chunk_t, axis=(1, 2), keepdims=True)) / \
                       (np.std(chunk_t, axis=(1, 2), keepdims=True) + 1e-4)

        # Search within [-20s, +40s] of current expected offset
        d_start = max(0, int((t_sec + expected_off - 20.0) * fps_sample))
        d_end = min(n_d - win_len, int((t_sec + expected_off + 40.0) * fps_sample))
        if d_end <= d_start:
            continue

        best_score, best_k = -1.0, d_start
        for k in range(d_start, d_end):
            score = np.mean(d_norm[k : k + win_len] * chunk_t_norm)
            if score > best_score:
                best_score, best_k = score, k

        offset_sec = best_k / fps_sample - t_sec
        if best_score >= 0.38:
            matches.append({"t": t_sec, "offset": offset_sec, "score": best_score})
            expected_off = 0.90 * expected_off + 0.10 * offset_sec

    return matches
```

**Key parameters and why:**
- `win_dur=4.0`: 4-second sequences are long enough to be unambiguous, short enough
  to not straddle scene cuts.
- `step_sec=1.0`: 1-second step gives dense coverage. Coarser steps (2s) work for
  a first pass but miss short scenes.
- `initial_offset`: Set this to a rough estimate (e.g. `80.0` if the donor has ~80s
  of extra intro logos). Get this from a single manual check in VLC.
- `score >= 0.38`: Empirically validated threshold. Below this, matches are unreliable.
  Above 0.50 is high confidence.
- `expected_off` tracking: The exponential moving average keeps the search window from
  wandering. Without it, the search window spans the full ±60s and picks up false matches.

#### Step 3.5: Native Frame-Rate Sub-Frame Calibration (The 0-Frame Refinement Pass)

> [!IMPORTANT]
> **The Coarse-Sampling Jitter Problem:**
> Extracting thumbnails at 2fps discretizes time into 500ms steps, introducing up to $\pm 250\text{ms}$ ($\pm 6\text{ frames}$ at 25fps) of sampling jitter. While sufficient to identify scene cuts, relying solely on raw 2fps offsets causes visible lip-sync lag (actors' mouth movements leading or trailing speech by 4–7 frames).
>
> **The Solution:** A rapid secondary pass at the film's **native frame rate (e.g. 25fps or 24fps)** on a single high-variance sequence inside each scene.

```python
import subprocess
import numpy as np

def calibrate_scene_offset_native_fps(target_video, donor_video, t_target, nominal_offset, fps=25.0):
    """
    Refines a macro offset to exact 0-frame peak precision (< 40ms error).
    Extracts 25 frames from target and 75 frames (±1s window) from donor at native framerate.
    """
    t_donor = t_target + nominal_offset
    win_len = int(fps * 0.8) # 20 frames = 0.8s sequence
    
    cmd_tgt = [
        'ffmpeg', '-y', '-ss', str(t_target), '-i', target_video,
        '-frames:v', str(int(fps)),
        '-vf', 'crop=iw*0.8:ih*0.8,scale=96:48,format=gray',
        '-f', 'rawvideo', '-'
    ]
    cmd_don = [
        'ffmpeg', '-y', '-ss', str(max(0.0, t_donor - 1.0)), '-i', donor_video,
        '-frames:v', str(int(fps * 3)),
        '-vf', 'scale=96:48,format=gray',
        '-f', 'rawvideo', '-'
    ]
    
    p1 = subprocess.Popen(cmd_tgt, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(cmd_don, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_tgt, _ = p1.communicate()
    raw_don, _ = p2.communicate()
    
    f_tgt = np.frombuffer(raw_tgt, dtype=np.uint8).reshape((-1, 96*48)).astype(np.float32)
    f_don = np.frombuffer(raw_don, dtype=np.uint8).reshape((-1, 96*48)).astype(np.float32)
    
    f_tgt_norm = (f_tgt - f_tgt.mean(axis=1, keepdims=True)) / (f_tgt.std(axis=1, keepdims=True) + 1e-4)
    f_don_norm = (f_don - f_don.mean(axis=1, keepdims=True)) / (f_don.std(axis=1, keepdims=True) + 1e-4)
    
    nominal_idx = int(fps) # Donor started 1.0s earlier
    best_score, best_shift = -1.0, 0
    
    for shift in range(-int(fps * 0.8), int(fps * 0.8) + 1): # ±20 frames search
        idx_don = nominal_idx + shift
        if 0 <= idx_don and idx_don + win_len <= len(f_don_norm):
            score = float(np.mean(f_tgt_norm[:win_len] * f_don_norm[idx_don : idx_don + win_len]))
            if score > best_score:
                best_score = score
                best_shift = shift
                
    # Refined offset in seconds
    refined_offset = nominal_offset + (best_shift / fps)
    return refined_offset, best_shift, best_score
```

- **Output**: Snaps the offset to the exact peak correlation ($r = 0.75 \to 0.965$).
- **Speed**: Calibrating 40+ scenes takes under **15 seconds** total.
- **Result**: Pin-to-pin, frame-perfect lip-sync matching professional audio director standards.

#### Step 4: Build Discrete Scene Map

Group consecutive matches with similar offsets into continuous scene blocks:

```python
def build_scene_map(matches, win_dur=4.0, max_gap=10.0, max_offset_diff=0.5):
    """
    Cluster matches into scenes. Bridge gaps between scenes.
    Returns list of (start_t, end_t, offset) covering the full timeline.
    """
    scenes = []
    curr = [matches[0]]

    for m in matches[1:]:
        if (abs(m["offset"] - curr[-1]["offset"]) <= max_offset_diff and
            (m["t"] - curr[-1]["t"]) <= max_gap):
            curr.append(m)
        else:
            if len(curr) >= 2:
                t0 = curr[0]["t"]
                t1 = curr[-1]["t"] + win_dur
                med_off = float(np.median([x["offset"] for x in curr]))
                scenes.append((t0, t1, med_off))
            curr = [m]

    if len(curr) >= 2:
        t0 = curr[0]["t"]
        t1 = curr[-1]["t"] + win_dur
        med_off = float(np.median([x["offset"] for x in curr]))
        scenes.append((t0, t1, med_off))

    # Bridge gaps so the entire timeline [0, end] is continuous
    # ... (fill gaps with nearest-neighbor offsets, merge identical-offset adjacents)

    return scenes
```

**Save the scene map to `tmp/scene_map.json`** for inspection and reuse.

#### Step 5: Assemble Synced Audio (Zero-Memory Streaming Assembly)

> **CRITICAL: DO NOT USE CONTINUOUS INTERPOLATION OR IN-MEMORY ARRAY ACCUMULATION.**
>
> 1. **Do not use continuous sample interpolation** (`interp1d`) — it causes Doppler flutter and metallic comb-filtering.
> 2. **Do not accumulate large arrays in memory** (`np.concatenate` in loops on multi-channel audio). For a 2.5-hour 6-channel FLAC, dynamic re-allocation consumes 20+ GB RAM, freezes the OS, and crashes background tasks.
>
> **The correct approach:** Use discrete integer-sample slicing with 20ms cosine crossfades, written via a **streaming block writer** directly to disk (`soundfile.SoundFile(..., 'w')`).

```python
import numpy as np
import soundfile as sf

def slice_and_stream(donor_path, out_flac_path, scene_map, target_duration_sec, sr=48000, xfade_samples=384):
    """
    Zero-memory streaming audio assembly with bit-perfect timeline preservation.
    Streams input FLAC scene-by-scene, applies micro-crossfades at boundaries in-place,
    and writes exactly (target_s1 - target_s0) samples per scene directly to disk with < 50MB RAM footprint.
    Zero samples lost, zero cumulative drift across the entire movie.
    """
    fade_in  = (np.sin(np.linspace(0, np.pi / 2, xfade_samples)) ** 2)[:, np.newaxis]
    fade_out = (np.cos(np.linspace(0, np.pi / 2, xfade_samples)) ** 2)[:, np.newaxis]

    total_target_samples = int(round(target_duration_sec * sr))
    total_written = 0

    with sf.SoundFile(donor_path, 'r') as f_in:
        num_frames = len(f_in)
        num_channels = f_in.channels

        with sf.SoundFile(out_flac_path, 'w', samplerate=sr, channels=num_channels, format='FLAC', subtype='PCM_24') as f_out:
            prev_tail = None

            for idx, sc in enumerate(scene_map):
                start_t = sc["start"]
                end_t = sc["end"]
                offset = sc["offset"]
                
                target_s0 = int(round(start_t * sr))
                target_s1 = int(round(end_t * sr))
                req_len = target_s1 - target_s0
                if req_len <= 0:
                    continue

                src_s0 = int(round((start_t + offset) * sr))
                src_s1 = src_s0 + req_len

                pad_pre = max(0, -src_s0)
                read_s0 = max(0, src_s0)
                read_s1 = min(num_frames, src_s1)
                read_count = max(0, read_s1 - read_s0)
                pad_post = max(0, req_len - (pad_pre + read_count))

                f_in.seek(read_s0)
                data = f_in.read(read_count, dtype='float32')
                if data.ndim == 1:
                    data = data[:, np.newaxis]

                chunks = []
                if pad_pre > 0:
                    chunks.append(np.zeros((pad_pre, num_channels), dtype='float32'))
                if len(data) > 0:
                    chunks.append(data)
                if pad_post > 0:
                    chunks.append(np.zeros((pad_post, num_channels), dtype='float32'))

                segment = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

                # Ensure segment length matches req_len bit-perfectly
                if len(segment) != req_len:
                    if len(segment) < req_len:
                        segment = np.pad(segment, ((0, req_len - len(segment)), (0, 0)))
                    else:
                        segment = segment[:req_len]

                # In-place micro-crossfade at boundary (zero samples dropped from timeline)
                if prev_tail is not None and len(segment) >= xfade_samples:
                    segment[:xfade_samples] = prev_tail * fade_out + segment[:xfade_samples] * fade_in

                # Save boundary tail for blending into next scene
                if idx < len(scene_map) - 1 and len(segment) >= xfade_samples:
                    prev_tail = segment[-xfade_samples:].copy()
                else:
                    prev_tail = None

                # Stream to disk in 64k sample blocks
                BLOCK_SIZE = 65536
                for b_start in range(0, len(segment), BLOCK_SIZE):
                    b_end = min(len(segment), b_start + BLOCK_SIZE)
                    f_out.write(segment[b_start:b_end])

                total_written += len(segment)

    print(f"Assembly complete! Total written: {total_written} samples (Expected: {total_target_samples}, Diff: {total_written - total_target_samples} samples)")
```

#### Step 6: Encode and Mux

```python
# Encode to AC-3 (Dolby Digital) preserving channel layout
ffmpeg -y -i "tmp/synced_audio_5ch.flac" -c:a ac3 -b:a 448k "tmp/synced_audio_5ch.ac3"

# Mux with untouched target video
mkvmerge -o "output/Title (Year) - 1080p AVC DD5.0 [Synced Audio].mkv" \
  --title "Title (Year)" \
  --video-tracks 0 --no-audio --no-subtitles \
  --track-name "0:1080p AVC" --language "0:tel" \
  "target.mkv" \
  --track-name "0:Telugu [DD 5.0 @ 448 kbps - DVD Synced]" \
  --language "0:tel" --default-track "0:yes" \
  "tmp/synced_audio_5ch.ac3" \
  --no-video --no-subtitles --audio-tracks 1 \
  --track-name "1:Hindi [AAC 2.0 @ 192 kbps - Dub]" \
  --language "1:hin" --default-track "1:no" \
  "target.mkv"

# Sanitize metadata
mkvpropedit "output/Title (Year) - 1080p AVC DD5.0 [Synced Audio].mkv" \
  --tags "all:" \
  --edit info --set "title=Title (Year)"
```

Use real ISO 639-2 language codes (`tel`, `hin`, `eng`, `tam`, `kan`, etc.).

---

### 2.2 Acoustic Correlation (Method B — Same-Language Only)

Use this when both sources have audio in the **same language** (e.g. syncing a DVD 5.1 Telugu track to a 1080p WebRip Telugu track).

> **DO NOT use acoustic correlation to sync audio between different languages.**
> It produces garbage offset maps. Use visual scene mapping (Section 2.1) instead.

#### Step 1: Extract 16kHz Mono References (Center Channel Focus)

When dealing with multi-channel donors (5.1/5.0), **do not average all channels into mono**. In a 5.1 film, dialogue is strictly isolated in the **Center Channel (Channel 2)**. Averaging surround channels and LFE rumble washes out speech transients and causes phase cancellation against a stereo WebRip mix.

```bash
# Target video (typically stereo AAC 2.0 -> 16kHz mono)
ffmpeg -y -i "target.mkv" -vn -map 0:a:0 -ac 1 -ar 16000 "tmp/ref_target_16k.wav"

# Donor video (extract Center channel if 5.1, or mono mixdown)
ffmpeg -y -i "donor.mkv" -vn -map 0:a:0 -af "pan=mono|c0=FC" -ar 16000 "tmp/ref_donor_16k.wav"
# (If donor is stereo, use standard -ac 1)
```

#### Step 2: Ultra-Fast Block RMS Energy Envelope (100 Hz Resolution)

Direct sample-by-sample waveform correlation fails between different masters because mastering compression, lossy codecs, and equalization alter waveform phase. **RMS energy envelopes** capture the temporal syllables and loudness contours identically regardless of EQ or compression.

Computing block RMS at 100 Hz (10 ms hop = 160 samples at 16 kHz) takes only ~0.5s for an entire 2.5-hour movie:

```python
import numpy as np
import soundfile as sf
from scipy import signal

HOP = 160 # 10 ms = 100 Hz
SR_ENV = 100

def compute_envelope(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    num_blocks = len(audio) // HOP
    # Vectorized block RMS: (num_blocks, 160) -> 1D RMS array
    blocks = audio[:num_blocks * HOP].reshape((num_blocks, HOP))
    rms = np.sqrt(np.mean(blocks**2, axis=1) + 1e-8)
    
    # 5-point Gaussian smoothing (50ms)
    smooth_kernel = np.array([0.061, 0.242, 0.383, 0.242, 0.061], dtype=np.float32)
    return signal.convolve(rms, smooth_kernel, mode="same")
```

#### Step 3: Dense Sliding Sequence Correlation

Correlate 10–12 second envelope chunks at every 1-2 second checkpoint across the movie. Use a two-tier search (Tier 1: ±30s local window around expected offset; Tier 2: broad `[-60s, +400s]` window if local score drops):

```python
# In NumPy, correlating a 1200-point chunk against a 6000-point slice takes 0.05ms.
# 4,500 checkpoints across 2.5 hours run in under 7 seconds!
# Score >= 0.50 yields frame-accurate (10ms resolution) offset matches.
```

#### Step 4: Clock Drift Analysis

Fit linear regression on anchor offsets:

$$\text{Offset}(t) = \text{Slope} \cdot t + \text{Base}$$

| Slope | Diagnosis | Action |
|:------|:----------|:-------|
| ~+0.001 s/s (+1000 ppm) | 23.976 vs 24.000 fps | Resample audio ×1000/1001 |
| ~-0.040 s/s | 25→24 fps PAL conversion | Resample audio ×24/25 |
| < 15 ppm | True 1:1 speed match | Fixed offset or scene-level slicing |

Confirm against `ffprobe`'s reported frame rates — don't rely on regression alone.

---

### 2.3 Hybrid Multi-Channel Stream Architecture (Uncut / Extended Master Handling)

A major hidden challenge when syncing legacy DVD9/Blu-ray 5.1 audio onto modern streaming WebRips (Zee5, SunNXT, Hotstar, Prime) is **Target-Exclusive Footage**:
- WebRips often contain uncut comedy tracks, extended dialogues, or alternate song segments (totaling 3–8 minutes) that were cut from physical DVD releases.
- **Never force donor audio across scenes that only exist in the target.** Doing so plays dialogue from completely wrong scenes and shifts all subsequent scene offsets by up to 90+ seconds, destroying second-half lip-sync.

#### Step 1: Detect Exclusive Scenes

During dense correlation (acoustic or visual), segments where the correlation score remains low ($< 0.35$) across the entire donor video for $> 6$ seconds indicate **WebRip-exclusive footage**.

#### Step 2: Prepare Target 5.1 Audio for Seamless Fill

Extract the target's reference audio upmixed to 6 channels so it matches the donor FLAC's layout and sample rate (dialogue stays anchored in Center channel):

```bash
ffmpeg -y -i "target.mkv" -vn -map 0:a:0 -af "surround=chl_out=5.1" -c:a flac "tmp/target_5ch_48k.flac"
```

#### Step 3: Build the Hybrid Scene Map

Create a gapless piecewise timeline from $0.000\text{s}$ to target duration:
- For donor-matched scenes: `source: "donor"`, `offset: calculated_offset`
- For target-exclusive scenes (and lead-in bumpers): `source: "target"`, `offset: 0.0`

#### Step 4: Stream and Crossfade

Stream directly from `donor_5ch_48k.flac` and `target_5ch_48k.flac` into the output FLAC, applying in-place 8ms micro-crossfades at source transitions:
- 90–95% of the film runs on the pristine DVD 5.1 master.
- Exclusive scenes seamlessly play their native WebRip audio.
- Zero audio dropped, zero mismatched dialogue, and 100% continuous lip-sync.

---

## 3. DVD Remuxing

### Extracting a Title from VIDEO_TS

```python
# 1. Identify the main title (longest duration)
ffprobe -v quiet -print_format json -show_format \
  "concat:VIDEO_TS/VTS_01_1.VOB|VIDEO_TS/VTS_01_2.VOB|..."

# 2. Lossless remux to MKV
mkvmerge -o "output/Title (Year) - 576p DVD MPEG2 AC3 5.0.mkv" \
  --title "Title (Year)" \
  --track-name "0:576p MPEG-2" --language "0:tel" \
  --track-name "1:Telugu [DD 5.0 @ 448 kbps]" --language "1:tel" \
  --chapter-language und \
  "VIDEO_TS/VTS_01_0.VOB" + "VIDEO_TS/VTS_01_1.VOB" + ...
```

Preserve chapters. Format chapter titles as `Chapter 01`, `Chapter 02`, etc.

---

## 4. Audio Extraction Guidelines

### Multi-Channel Audio (DVD/Blu-ray 5.0/5.1)

Extract to FLAC (lossless, no 4GB WAV limit):

```bash
ffmpeg -y -i "source.mkv" -vn -map 0:a:0 -c:a flac "tmp/donor_5ch_48k.flac"
```

> **WARNING: Do NOT extract multi-channel audio to WAV.**
> 5-channel 48kHz audio for a 2.5-hour movie exceeds the 4GB WAV file size limit
> (32-bit header). The file will be silently truncated. Use FLAC instead.

### Stereo Audio

WAV is fine for stereo (< 4GB for typical movie lengths):

```bash
ffmpeg -y -i "source.mkv" -vn -map 0:a:0 -ac 2 -ar 48000 "tmp/donor_stereo_48k.wav"
```

---

## 5. Video Deinterlacing & Combing Removal

### Detection

$$\text{Comb}(x, y) = |2 \cdot I(x, y) - I(x, y-1) - I(x, y+1)|$$

Values > 20 with motion peaks well above indicate baked-in interlacing.

### Field Parity (TFF vs BFF)

```bash
ffmpeg -ss 180 -i "source.mkv" -vf "bwdif=mode=send_frame:parity=tff:deint=all" \
  -frames:v 1 "tmp/tff_test.png"
ffmpeg -ss 180 -i "source.mkv" -vf "bwdif=mode=send_frame:parity=bff:deint=all" \
  -frames:v 1 "tmp/bff_test.png"
```

Keep whichever parity scores lower comb energy.

### Filters

```bash
# BWDIF (recommended default)
ffmpeg -i "input.mkv" \
  -vf "bwdif=mode=send_frame:parity=auto:deint=all" \
  -c:v libx264 -preset fast -crf 18 -c:a copy "output/deinterlaced.mkv"

# IVTC (29.97i → 23.976p, 3:2 pulldown)
ffmpeg -i "input.mkv" \
  -vf "fieldmatch=order=auto:combmatch=full,decimate=cycle=5" \
  -c:v libx264 -preset fast -crf 18 -c:a copy "output/ivtc.mkv"
```

---

## 6. File Naming & Metadata Hygiene

### Naming Convention

`Title (Year) - [Resolution] [VideoCodec] [AudioCodec Channels] [Notes].mkv`

Examples:
- `Title (Year) - 1080p AVC DD5.0 [Synced Audio].mkv`
- `Title (Year) - 576p DVD MPEG2 AC3 5.0.mkv`
- `Title (Year) - 720p AVC AAC [Deinterlaced BWDIF].mkv`

**No "remaster", "master", "AI", or quality-claim language.** This pipeline synchronizes
and remuxes — it does not perform authoritative studio remastering.

### Metadata Sanitization

Strip uploader branding, tracker URLs, and promotional text. Set clean titles and real
ISO 639-2 language codes:

```bash
mkvpropedit "output/file.mkv" \
  --tags "all:" \
  --edit info --set "title=Title (Year)"
```

---

## 7. Verification Checklist

- [ ] **Video untouched:** target video stream byte-identical to source (not re-encoded,
      not split, not cut).
- [ ] **Audio sounds natural:** no robotic/metallic artifacts, no pitch shifting, no
      flutter. Play back the synced audio and confirm it sounds identical to the original
      donor audio.
- [ ] **Lip sync verified:** spot-check at least 5 points across the timeline (beginning,
      25%, 50%, 75%, end) — dialogue matches lip movements.
- [ ] **Automated multi-point cross-correlation:** extract 12-second snippets of Track 1
      (synced audio) against Track 2 (reference audio) across 8–11 checkpoints across the entire film.
      Verify that relative delay is within $\pm 40\text{ ms}$ ($< 1$ video frame) and envelope
      correlation $r \ge 0.70$.
- [ ] **Channel layout correct:** multi-channel audio has correct channel order
      (`FL, FR, FC, LFE, SL, SR` for 5.1; `FL, FR, FC, SL, SR` for 5.0).
- [ ] **No silence gaps:** no unexpected silence at scene boundaries. Crossfades are
      smooth and inaudible.
- [ ] **Duration match:** synced audio duration matches target video duration within
      ±100ms.
- [ ] **File naming:** matches the plain convention in Section 6.
- [ ] **Metadata clean:** no tracker/uploader branding in container or stream tags.
- [ ] **Language tags correct:** real ISO 639-2 codes (`tel`, `hin`, `eng`, etc.), not
      `und` placeholders.
- [ ] **Workspace hygiene:** all scratch files under `tmp/`. Only verified deliverables
      in `output/`.

---

## 8. Known Failure Modes

These are documented mistakes that produce bad output. Do not repeat them.

### ❌ Forcing Donor Audio Across Target-Exclusive / Uncut Scenes

**What it is:** Assuming the donor audio has content for every second of the target video, and forcibly stretching or shifting donor scenes over uncut streaming footage.

**Why it fails:** Modern streaming WebRips (Zee5, SunNXT, Hotstar, Prime) often contain uncut comedy tracks, extended dialogues, or alternate song segments (totaling 3–8 minutes) that were cut from physical DVD releases. Forcing donor audio across these missing scenes creates severe desynchronization (up to 90+ seconds) and places dialogue from completely wrong scenes onto the screen, ruining the entire second half of the film.

**Correct approach:** Use the Hybrid Multi-Channel Stream Architecture (Section 2.3). Detect exclusive scenes via continuous low correlation ($< 0.35$ for $> 6$s) and seamlessly patch those segments with the native target audio upmixed to 5.1.

### ❌ Cumulative Sample Loss in Streaming Crossfade Buffers

**What it is:** Holding back `xfade_samples` (e.g. 20ms = 960 samples) at each scene boundary without compensating on the read side, so that each scene outputs `req_len - xfade_samples` samples.

**Why it fails:** While 20ms seems negligible for a single transition, over 100+ scene cuts across a 2.5-hour film, losing 20ms per cut accumulates into an artificial linear clock drift of $2.0 \to 3.0\text{ seconds}$ by the end of the film.

**Correct approach:** Enforce bit-perfect sample math (`target_s1 - target_s0`). Perform crossfading in-place on the boundary samples of each chunk without dropping timeline samples, guaranteeing that total written samples equals $\text{round}(\text{duration} \times \text{sr})$ with $0\text{ samples}$ difference.

### ❌ Averaging All 5.1 Channels Into Mono for Acoustic Correlation

**What it is:** Downmixing 5.1/5.0 audio by taking the mean of all 6 channels (`np.mean(chunk, axis=1)`) when generating 16kHz mono reference files.

**Why it fails:** In film mixing, dialogue is isolated in the Center channel (Channel 2), while surround channels and LFE contain out-of-phase ambient effects, music reverb, and low-frequency rumble. Averaging all channels causes phase cancellation and drowns out speech transients, reducing cross-correlation peaks from $0.85+$ down to $< 0.05$.

**Correct approach:** Extract the Center channel (`pan=mono|c0=FC`) for 5.1 tracks before computing RMS energy envelopes.

### ❌ Continuous Sample-by-Sample Interpolation

**What it is:** Using `scipy.interpolate.interp1d` to create a continuous time-warping
function and resampling every audio sample through it.

**Why it fails:** Micro-fluctuations in the interpolated offset function (caused by
noisy correlation anchors) continuously pitch-shift the audio by tiny amounts. This
creates Doppler flutter and metallic comb-filtering. The audio sounds "robotic."

**Correct approach:** Integer-sample slicing with discrete scene blocks (Section 2.1
Step 5).

### ❌ Acoustic Correlation Between Different Languages

**What it is:** Cross-correlating Hindi audio against Telugu audio (or any two different
languages) to find timing offsets.

**Why it fails:** Different languages have completely different speech patterns, phonemes,
and timing. The correlation scores are low and noisy, producing offset maps that jump
randomly between false matches. Background music provides some signal, but it's too weak
and ambiguous for reliable alignment.

**Correct approach:** Visual frame matching (Section 2.1 Steps 2–3), which is
language-independent.

### ❌ In-Memory Multi-Channel Audio Accumulation
 
**What it is:** Concatenating multi-gigabyte numpy arrays in loops (`master = np.concatenate([master, nxt])`) across 30+ scenes.
 
**Why it fails:** For a 2.5-hour 6-channel 48kHz FLAC, each re-allocation consumes 10-20 GB of RAM. This causes immediate Windows pagefile thrashing, freezes the user's desktop, and crashes background tasks with Out of Memory errors.
 
**Correct approach:** Zero-memory streaming assembly with `soundfile.SoundFile` streaming writer directly to disk (Section 2.1 Step 5).
 
### ❌ Extracting Multi-Channel Audio to WAV

**What it is:** Using `ffmpeg -f wav` to extract 5+ channel audio from long movies.

**Why it fails:** WAV uses a 32-bit header, limiting files to 4GB. A 2.5-hour 5-channel
48kHz 24-bit file is ~6.4GB. The file gets silently truncated.

**Correct approach:** Extract to FLAC (`-c:a flac`).

### ❌ Single-Frame Visual Matching

**What it is:** Correlating individual frames between target and donor.

**Why it fails:** Many frames in a movie look visually similar (dark scenes, wide shots,
closeups of the same actor). Single-frame matching produces many false positives.

**Correct approach:** Multi-frame sequence correlation (4–6 second windows). A sequence
of frames is much more unique than any individual frame.

### ❌ Subtitle Distributor Watermark Pollution & Donor Drift

**What it is:** Directly time-shifting raw donor subtitle rips without stripping distributor watermarks or checking for intra-scene pacing differences.

**Why it fails:** Donor DVDs often contain injected 10-second watermark cards (e.g. `SRI BALAJI`, `EAGLE VIDEO`) that overwrite dialogue lines. Furthermore, old telecine DVD subtitle rips frequently suffer from inconsistent sentence spacing and clock drift that standard macro offsets cannot fully resolve.

**Correct approach:** Strip all watermark cues, perform VAD/acoustic speech onset locking, or use Whisper on GPU for direct speech-synchronized translation (Section 10.3).

### ❌ Assuming Identical Inter-Stream Container Timecodes

**What it is:** Assuming all audio tracks inside the same donor MKV (e.g. Track 1: AC3, Track 2: DTS) share the same starting timecode and zero relative offset.

**Why it fails:** DVD/Blu-ray container remuxes frequently have internal stream delays (e.g. DTS starting +10.84s later than AC3 in the container). Reusing the offset map calculated on one stream for another stream without measuring the inter-stream delay produces severe, constant desynchronization on the secondary audio track.

**Correct approach:** Always extract reference audio for EACH distinct audio stream being synchronized, or calculate the exact cross-correlation offset between donor tracks before applying scene maps.

### ❌ Ignoring Integer vs Drop-Frame Clock Drift (23.976 vs 24.000 fps)

**What it is:** Slicing audio without testing for the $1000/1001$ film pulldown ratio when matching DVD (23.976 fps) audio to Clean Master (24.000 fps) video.

**Why it fails:** The $1000/1001$ speed ratio introduces a constant cumulative drift of $+1.001001\text{ ms/s}$ ($+3.6\text{ s/hour}$). Over a 2.5-hour film, this creates an $+8.74\text{-second}$ progressive drift that breaks lip-sync inside every continuous scene block.

**Correct approach:** Always perform linear regression on anchor offsets ($\text{Slope} \approx +0.001\text{ s/s}$). Resample donor audio losslessly with `asetrate=48048,aresample=48000` (or `atempo`) BEFORE discrete scene assembly.

### ❌ Relying Solely on Coarse 2fps Offsets (Lip-Sync Jitter)

**What it is:** Using the raw offset values directly from 2fps visual thumbnails without a native framerate refinement pass.

**Why it fails:** At 2fps, each sample step is 500ms. Discretization jitter introduces errors up to $\pm 250\text{ms}$ ($\pm 6\text{ frames}$ at 25fps). While good enough for establishing scene cuts, the audio will perceptibly lead or trail dialogue by 4–7 frames, breaking professional lip-sync.

**Correct approach:** Always execute the native framerate sub-frame calibration pass (Section 2.1 Step 3.5). Extract 25 frames from target and 75 frames from donor at native framerate (24/25fps) across candidate shifts $\pm 20$ frames to discover the exact 0-frame peak correlation ($r \ge 0.85$).

---

## 9. Framerate Conversion Reference

| Conversion | Purpose | Video Filter | Audio Adjustment |
|:-----------|:--------|:-------------|:-----------------|
| 23.976 → 24.000 fps | Pull-up | `setpts=(1000/1001)*PTS`, `-r 24` | Resample ×1001/1000 |
| 24.000 → 23.976 fps | Pull-down | `setpts=(1001/1000)*PTS`, `-r 24000/1001` | Resample ×1000/1001 |
| 24.000 → 25.000 fps | PAL speedup | `setpts=(24/25)*PTS`, `-r 25` | `atempo=25/24` |
| 25.000 → 24.000 fps | PAL slowdown | `setpts=(25/24)*PTS`, `-r 24` | `atempo=24/25` |

---

## 10. Subtitle Sync, OCR Repair & Whisper AI Alignment

### 10.1 Subtitle Extraction & Watermark Sanitization

1. **Detect encoding:** Handle UTF-8, UTF-16 LE/BE, and null-byte-delimited text.
2. **OCR Repair:**
   - `\bl\b` → `I`, `\blt\b` → `It`, `\bls\b` → `Is`, `\blf\b` → `If`
   - `\bl'm\b` → `I'm`, `\bl'll\b` → `I'll`, `\bl've\b` → `I've`
   - Spaced OCR (`T h i s`) → collapsed (`This`)
3. **Watermark & Spam Stripping:**
   - Detect and remove all hardcoded distributor watermarks (e.g. `SRI BALAJI`, `EAGLE VIDEO`, `AD`, `SUBSCRIBE`, anti-piracy cards).

### 10.2 Time-Warping Subtitle Timestamps (Method A)

When donor subtitles are structurally intact:
1. Apply the framerate pulldown/pull-up factor ($1000/1001$ for $23.976 \to 24.000$ fps).
2. Look up the corresponding scene block in `scene_map.json`:
   $$t_{\text{target}} = t_{\text{donor\_pulldown}} - \text{scene.offset}$$

### 10.3 (Optional) AI Speech-Locked Subtitle Generation via Whisper on GPU (Method B)

If donor subtitles suffer from severe intra-scene drift, missing lines, or unfixable telecine authoring defects, use **Whisper on GPU** for direct neural translation:

```python
import whisper
import re

def generate_whisper_subtitles(ref_target_wav, out_srt_path):
    # Load Whisper medium or large-v3 on GPU
    model = whisper.load_model("medium", device="cuda")
    
    # Transcribe & Translate directly from target audio
    result = model.transcribe(
        ref_target_wav,
        task="translate",
        language="Telugu",  # Or source language
        fp16=True,
        temperature=0.0,
        condition_on_previous_text=False
    )
    
    # Post-process: deduplicate consecutive phrases and format SRT
    # ...
```

**Key Advantages:**
- **Zero Drift:** Subtitle cues are generated directly from the target video's audio timeline.
- **Speech-Locked:** Every cue starts exactly when the actor begins speaking and ends when they finish.
- **Zero Watermarks:** Eliminates all legacy DVD watermark pollution.

---

## 11. HDTV Source De-Advertising & Lossless Ad-Pod Removal

When processing HDTV broadcasts (e.g. JioTV+, Zee Cinemalu, Gemini TV, Star Maa, ETV), full commercial break pods (typically 7–10 pods, 3–6 minutes each, totaling 30–45+ minutes) are inserted into the video stream. To produce a clean, uncut movie file without re-encoding, follow this lossless pipeline:

### 11.1 Broadcast Marker Analysis

Indian HDTV film broadcasts exhibit four distinct on-screen indicators during commercials:
1. **Letterbox Black Bar Blowout:** During the film, scope letterboxing (2.35:1) keeps the top and bottom rows (`y: 0–15` and `y: 1065–1080`) at mean luminance $0.0$. Full-frame 16:9 commercials blow out letterbox rows to $> 15.0$ mean luminance.
2. **"Back in MM:SS" Countdown Overlays:** For letterboxed commercials, channels overlay a progress line and countdown timer (e.g. `Back in 01:15`) in the top-left letterbox area (`y: 105–115, x: 50–250`).
3. **Channel Entry Bumpers:** Commercial break entry is marked by a 3D animated channel bumper (e.g. Zee Cinemalu "Dil Pai...", Star Maa ident).
4. **"Now Playing" Return Banners:** When the movie resumes from an ad break, the network displays a "Now Playing [Title]" banner in the top-left corner for 10–15 seconds (`y: 105–115, x: 50–250`, mean luminance $\approx 58–60$).

### 11.2 Acoustic Timeline Cross-Correlation

To isolate movie footage from advertisements across the 3-hour recording:
1. Extract 16 kHz mono reference audio from both the HDTV file and an uncut clean master (DVD/WEB-DL).
2. Compute 100 Hz Block RMS energy envelopes ($10\text{ ms}$ hop size).
3. Sliding window correlation ($12\text{s}$ windows every $2\text{s}$):
   - **Movie:** $r \ge 0.50$ (audio matches the clean film master).
   - **Commercial Break:** $r < 0.35$ (commercial jingles/ads have zero correlation with the film master).
4. Any timeline offset jump ($\Delta \text{offset} > 10\text{s}$) indicates a commercial break or broadcast scene cut.

### 11.3 Lossless Stream Concatenation via MKVMerge

To remove advertisements **without re-encoding** (preserving 100% untouched AVC/HEVC video and AAC audio bit-for-bit):
1. Assemble the exact time ranges of all movie segments into `mkvmerge --split parts:...` syntax:
   ```bash
   "C:\Program Files\MKVToolNix\mkvmerge.exe" -o "output/Movie (Year) [Clean - No Ads].mkv" \
     --split parts:00:00:00.000-00:16:45.920,+00:22:00.920-00:31:27.640,+00:36:12.640-00:46:08.440,... \
     "input_hdtv.mkv"
   ```
   *(Note: The `+` prefix before subsequent parts instructs `mkvmerge` to append them into a single continuous file rather than splitting into multiple files).*
2. Splice execution takes $< 10\text{ seconds}$ for a 4.5 GB stream with zero quality loss.
3. Sanitize container metadata with `mkvpropedit`:
   ```bash
   mkvpropedit "output/Movie (Year) [Clean - No Ads].mkv" \
     --edit info --set "title=Movie (Year) [Clean - No Ads]" \
     --edit track:v1 --set "name=1080p HDTV AVC [Clean - No Ads]" \
     --edit track:a1 --set "name=Telugu [AAC 2.0]" --set "language=tel"
   ```

---

## 12. Lossless Intro / Head-Tail Trimming via Keyframe Splitting

When an assembled or remastered release contains unwanted distributor logos, regional title cards, or censor certificates preceding the film (e.g. 75s of black screens/titles):

### Step 1: Detect the Exact Boundary Keyframe (I-Frame)

```bash
ffprobe -v error -select_streams v:0 \
  -show_frames -read_intervals 70%80 \
  -show_entries frame=pts_time,pict_type,key_frame \
  -of json "output/file.mkv"
```

Locate the keyframe (`key_frame: 1`) immediately at or before the transition from titles to film picture (e.g. `pts_time: 75.000000, key_frame: 1` corresponding to `00:01:15.000`).

### Step 2: Losslessly Trim Container with Zero Re-Encoding

```bash
mkvmerge -o "output/file.trimmed.mkv" \
  --split parts:00:01:15- \
  "output/file.mkv"
```

- **Zero Re-encoding:** Video and audio streams are copied byte-for-byte.
- **Zero Drift:** All audio tracks (5.1 surround, stereo, dub tracks) are split at the identical timestamp, preserving perfect lip-sync from the very first frame of the trimmed file.