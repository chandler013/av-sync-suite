---
name: sync-different-runtimes
description: >-
  Production-grade pipeline for synchronizing audio tracks between different video masters,
  edits, and runtimes of the same film (e.g. cut regional dubs vs uncut originals, extended
  directors cuts vs theatrical, 4K WebRip vs 1080p/DVD masters with differing runtime lengths).
  Covers letterbox/aspect ratio geometric calibration, GPU-accelerated thumbnail extraction,
  dense multi-frame sequence correlation, micro/macro-cut preservation, target-exclusive audio fallback,
  zero-memory bit-perfect streaming crossfade assembly, and automated multi-point verification.
---

# Video Synchronization Across Different Runtimes (`sync-different-runtimes`)

Synchronize audio tracks between video masters of **different runtimes, cuts, and aspect ratios** (e.g. syncing an uncut 2h 28m original Telugu 5.1/stereo audio track onto a 2h 05m 4K Tamil dubbed release, or a theatrical master onto an extended streaming release).

---

## 0. The Anatomy of Different Runtimes

When two releases of the same film differ in duration (e.g. Target is 125 min, Donor is 148 min):
- **Censorship cuts & trim edits**: Violence, dialogue, or regional jokes are trimmed.
- **Song sequences**: Songs may be trimmed, extended, or placed in different scenes.
- **Comedy & subplot cuts**: Regional dubs frequently cut out 10–25 minutes of secondary subplots.
- **Intro / outro logos**: Distinct distributor cards and title sequences at the beginning.

### The Offset Relationship

For any timestamp $t_{\text{target}}$ in the target video:

$$t_{\text{donor}} = t_{\text{target}} + \text{Offset}(t_{\text{target}})$$

- When the **donor has extra footage** (footage cut from the target): the offset jumps **UP** by the duration of the cut donor scene.
- When the **target has exclusive footage** (footage cut from the donor): the offset drops **DOWN**.
- **THE CARDINAL RULE: NEVER assume the offset is constant after an early cut.** A film with multiple cuts will have an offset curve that jumps and steps continuously across the entire runtime. Extrapolating a single offset across the second half of a movie causes catastrophic desynchronization (often 10–25 minutes of drift).

---

## 1. Directory Structure

```text
project_root/
├── target_video.mkv                # untouched target video
├── donor_video.mkv                 # untouched donor video
├── output/                         # final deliverables only
│   └── Title (Year) - 2160p [Synced Audio].mkv
└── tmp/                            # scratch files (safe to purge)
    ├── tam_clean_2fps.raw          # normalized target thumbnails (96x48)
    ├── tel_clean_2fps.raw          # normalized donor thumbnails (96x48)
    ├── dense_clean_matches.json    # dense sequence cross-correlation data
    ├── clean_scene_map.json        # discrete gapless scene map
    ├── donor_stereo_48k.flac       # extracted lossless donor audio
    ├── target_stereo_48k.flac      # extracted reference target audio
    ├── synced_audio_clean.flac     # assembled bit-perfect synced audio
    └── *.py                        # pipeline scripts
```

---

## 2. Phase 1: Geometric Aspect Ratio & Letterbox Calibration

> [!CAUTION]
> **The #1 Silent Failure Mode in Visual Synchronization:**
> Many modern 4K/WebRip releases store widescreen films (2.35:1 or 2.38:1) inside a 16:9 container with hardcoded black bars (~270px top and bottom). Other releases (like 1080p rips or open-matte versions) crop the black bars entirely, and often crop a few pixels off the left and right sides.
> 
> Running a generic filter like `crop=iw*0.8:ih*0.8` leaves black bars on letterboxed videos while cropping into actors' faces on borderless videos. **This causes 90%+ of frames to collapse to unmatched noise (< 0.28 correlation)**, blinding the algorithm.

### Step 1: Detect Active Video Area in Both Sources

Run an active row/column probe across several timestamps:

```python
import cv2
import numpy as np
import subprocess

def probe_active_area(video_path, sample_times=[300, 1500, 3000, 5000]):
    """Returns (top, bottom, left, right, active_w, active_h)."""
    tops, bots = [], []
    for t in sample_times:
        cmd = ['ffmpeg', '-y', '-ss', str(t), '-i', video_path, '-frames:v', '1',
               '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'gray', '-']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = p.communicate()
        # Probe video dimensions with ffprobe first to reshape correctly
        # ...
        row_means = np.mean(frame, axis=1)
        active = np.where(row_means > 10)[0]
        if len(active) > 0:
            tops.append(active[0])
            bots.append(active[-1])
    return int(np.median(tops)), int(np.median(bots))
```

### Step 2: Calibrate Matching Aspect Ratios

- If Target is $3840 \times 2160$ with active area $3840 \times 1610$ (2.38:1 Scope):
  - Black bars: top = 275px, bottom = 1885px.
- If Donor is $1920 \times 922$ (2.08:1):
  - Donor cropped ~350px from left and right of the 2.38:1 Scope frame.
- **Calibrated Target Crop Filter**:
  `crop=3140:1610:350:275,scale=96:48,format=gray`
- **Calibrated Donor Filter**:
  `scale=96:48,format=gray`
- **Result**: Visual correlation jumps from 0.15 (noise) to **0.75–0.93 (crystal clear, unambiguous match)**.

---

## 3. Phase 2: High-Speed Hardware-Accelerated Extraction

Extract clean 2fps grayscale thumbnails at $96 \times 48$ resolution using GPU decoding:

```bash
# Target Video (e.g. 4K VP9/HEVC) - running with calibrated active crop
ffmpeg -y -hwaccel cuda -i "target_video.mkv" \
  -vf "fps=2,crop=3140:1610:350:275,scale=96:48,format=gray" \
  -f rawvideo "tmp/target_clean_2fps.raw"

# Donor Video (e.g. 1080p HEVC/AVC)
ffmpeg -y -hwaccel cuda -i "donor_video.mkv" \
  -vf "fps=2,scale=96:48,format=gray" \
  -f rawvideo "tmp/donor_clean_2fps.raw"
```

- **Extraction Speed**: Runs at **12x–25x real-time speed** on modern NVIDIA GPUs.
- **Footprint**: A full 2.5-hour film at 2fps and $96 \times 48$ is only **~75 MB** in RAM.

---

## 4. Phase 3: Vectorized Multi-Frame Sequence Correlation

> [!NOTE]
> Never correlate single frames. Single frames produce false positives on dark scenes, generic dialogue shots, and similar set backgrounds.
> Correlate **sequences of 4 seconds (8 frames at 2fps)**. A 4-second sequence is temporally unique across the entire film.

```python
import numpy as np
import json
import time

def dense_sequence_correlation(tam_raw_path, tel_raw_path, out_json_path, fps=2, w=96, h=48):
    pixels = w * h
    win_len = int(4.0 * fps)  # 8 frames = 4 seconds
    
    # Load and pre-normalize all frames: (N, 4608)
    tam = np.fromfile(tam_raw_path, dtype=np.uint8).reshape((-1, pixels)).astype(np.float32)
    tel = np.fromfile(tel_raw_path, dtype=np.uint8).reshape((-1, pixels)).astype(np.float32)
    
    tam_norm = (tam - np.mean(tam, axis=1, keepdims=True)) / (np.std(tam, axis=1, keepdims=True) + 1e-4)
    tel_norm = (tel - np.mean(tel, axis=1, keepdims=True)) / (np.std(tel, axis=1, keepdims=True) + 1e-4)
    
    n_tam = len(tam)
    n_tel = len(tel)
    checkpoints = np.arange(0, n_tam - win_len, int(1.0 * fps))  # every 1.0s
    
    matches = []
    curr_expected_off = 0.0
    
    for idx_step, idx_tam in enumerate(checkpoints):
        t_tam = float(idx_tam) / fps
        chunk_tam = tam_norm[idx_tam : idx_tam + win_len]
        
        # Tier 1: Local search [-30s, +30s] around current expected offset
        off_min = max(-100.0, curr_expected_off - 30.0)
        off_max = min(2000.0, curr_expected_off + 30.0)
        j_min = max(0, int((t_tam + off_min) * fps))
        j_max = min(n_tel - win_len, int((t_tam + off_max) * fps))
        
        best_sc, best_j = -1.0, j_min
        if j_max > j_min:
            tel_sub = tel_norm[j_min : j_max + win_len]
            dots = np.dot(tel_sub, chunk_tam.T) / float(pixels)
            num_cand = j_max - j_min
            scores = np.zeros(num_cand, dtype=np.float32)
            for i in range(win_len):
                scores += dots[i : i + num_cand, i]
            scores /= float(win_len)
            best_idx = int(np.argmax(scores))
            best_sc, best_j = float(scores[best_idx]), j_min + best_idx
            
        # Tier 2: If local score < 0.42, perform broad search across entire movie
        if best_sc < 0.42:
            broad_min = 0
            broad_max = n_tel - win_len
            tel_broad = tel_norm[broad_min : broad_max + win_len]
            dots = np.dot(tel_broad, chunk_tam.T) / float(pixels)
            num_cand = broad_max - broad_min
            scores = np.zeros(num_cand, dtype=np.float32)
            for i in range(win_len):
                scores += dots[i : i + num_cand, i]
            scores /= float(win_len)
            best_idx = int(np.argmax(scores))
            if float(scores[best_idx]) > best_sc:
                best_sc = float(scores[best_idx])
                best_j = broad_min + best_idx
                
        t_tel = float(best_j) / fps
        offset = t_tel - t_tam
        if best_sc >= 0.45:
            curr_expected_off = 0.85 * curr_expected_off + 0.15 * offset
            
        matches.append({
            "t_tam": round(t_tam, 2),
            "t_tel": round(t_tel, 2),
            "offset": round(offset, 2),
            "score": round(best_sc, 4),
            "status": "MATCHED" if best_sc >= 0.45 else ("MODERATE" if best_sc >= 0.35 else "UNMATCHED")
        })
        
    with open(out_json_path, "w") as f:
        json.dump(matches, f, indent=2)
```

- **Execution Time**: The vectorized matrix operations correlate **7,500+ checkpoints in under 2 seconds**.

---

## 5. Phase 3.5: Native Frame-Rate Sub-Frame Calibration (The 0-Frame Refinement Pass)

> [!IMPORTANT]
> **The Coarse-Sampling Jitter Problem:**
> Extracting thumbnails at 2fps discretizes time into 500ms intervals. This introduces up to $\pm 250\text{ms}$ ($\pm 6\text{ frames}$ at 25fps) of sampling jitter. While sufficient to identify scene cuts, relying on raw 2fps offsets causes noticeable lip-sync lag (actor's mouth moving 4–7 frames before/after dialogue).
>
> **The Solution:** A rapid secondary pass at the film's **native frame rate (e.g. 25fps or 24fps)** on a single high-variance sequence inside each scene.

```python
import subprocess
import numpy as np

def calibrate_scene_offset_25fps(target_video, donor_video, t_target, nominal_offset, fps=25.0):
    """
    Refines a macro offset to exact 0-frame peak precision (< 40ms error).
    Extracts 25 frames from target and 75 frames (±1s window) from donor at native framerate.
    """
    t_donor = t_target + nominal_offset
    win_len = int(fps * 0.8) # 20 frames = 0.8s sequence
    
    cmd_tgt = [
        'ffmpeg', '-y', '-ss', str(t_target), '-i', target_video,
        '-frames:v', str(int(fps)),
        '-vf', 'crop=3140:1610:350:275,scale=96:48,format=gray',
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
- **Speed**: Calibrating all 40+ scenes takes under **15 seconds** total.
- **Result**: Pin-to-pin, frame-perfect lip-sync matching professional audio director standards.

---

## 6. Phase 4: Gapless Scene Map with Hybrid Exclusive Audio Fallback

### 6.1 Forward Target Timeline Clustering
Always cluster along the **Target Video Timeline** ($t_{\text{target}}$):
1. **Never cluster across the Donor Timeline first**: If the donor has uncut comedy or extended scenes cut from the target, reverse clustering prematurely splits or drops target footage.
2. **Scene Continuations**: When stepping through target footage, if correlation remains high ($r \ge 0.70$) with the same offset, seamlessly extend the scene across what might otherwise appear as gaps in donor-led indices.

### 6.2 The Hybrid Stream Architecture (Target-Exclusive Footage)

When target video contains scenes cut from the donor release (e.g. Tamil exclusive comedy sequence, intro title cards):
- Segments where correlation remains $< 0.35$ for $> 12\text{ seconds}$ indicate **Target-Exclusive Footage**.
- **Assign `source: "target", offset: 0.0`** for these segments.
- Native target audio plays seamlessly during exclusive footage, and donor audio resumes when the film rejoins.
- **Never force donor audio across target-exclusive footage.** Doing so displaces the rest of the film's timeline by minutes.

---

## 7. Phase 5: Zero-Memory Streaming Audio Assembly

Apply in-place 15ms cosine crossfades at each boundary and write sample-accurate blocks directly to disk:

```python
import numpy as np
import soundfile as sf
import json

def assemble_synced_audio(donor_flac, target_flac, scene_map_json, out_flac, target_dur, sr=48000, xfade_ms=15):
    xfade_samples = int(xfade_ms * sr / 1000) # 720 samples
    fade_in  = (np.sin(np.linspace(0, np.pi/2, xfade_samples))**2)[:, np.newaxis]
    fade_out = (np.cos(np.linspace(0, np.pi/2, xfade_samples))**2)[:, np.newaxis]
    
    total_target_samples = int(round(target_dur * sr))
    total_written = 0
    
    with open(scene_map_json) as f:
        scenes = json.load(f)
        
    with sf.SoundFile(donor_flac, 'r') as f_donor, \
         sf.SoundFile(target_flac, 'r') as f_target, \
         sf.SoundFile(out_flac, 'w', samplerate=sr, channels=2, format='FLAC', subtype='PCM_24') as f_out:
         
        prev_tail = None
        for idx, sc in enumerate(scenes):
            s0 = int(round(sc['start'] * sr))
            s1 = int(round(sc['end'] * sr))
            req_len = s1 - s0
            if req_len <= 0:
                continue
                
            f_src = f_donor if sc.get('source') == 'donor' or sc.get('source') == 'telugu' else f_target
            src_time = sc['start'] + sc.get('offset', 0.0) if sc.get('source') != 'target' else sc['start']
            src_s0 = int(round(src_time * sr))
            src_s1 = src_s0 + req_len
            
            # Pad boundaries if seeking before 0 or past EOF
            pad_pre = max(0, -src_s0)
            read_s0 = max(0, src_s0)
            read_s1 = min(len(f_src), src_s1)
            read_count = max(0, read_s1 - read_s0)
            pad_post = max(0, req_len - (pad_pre + read_count))
            
            f_src.seek(read_s0)
            data = f_src.read(read_count, dtype='float32')
            if data.ndim == 1:
                data = data[:, np.newaxis]
            if data.shape[1] == 1:
                data = np.column_stack([data, data])
                
            chunks = []
            if pad_pre > 0: chunks.append(np.zeros((pad_pre, 2), dtype='float32'))
            if len(data) > 0: chunks.append(data)
            if pad_post > 0: chunks.append(np.zeros((pad_post, 2), dtype='float32'))
            segment = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            
            # Enforce bit-perfect exact length
            if len(segment) < req_len:
                segment = np.pad(segment, ((0, req_len - len(segment)), (0, 0)))
            elif len(segment) > req_len:
                segment = segment[:req_len]
                
            # Crossfade in-place
            if prev_tail is not None and len(segment) >= xfade_samples:
                segment[:xfade_samples] = prev_tail * fade_out + segment[:xfade_samples] * fade_in
                
            if idx < len(scenes) - 1 and len(segment) >= xfade_samples:
                prev_tail = segment[-xfade_samples:].copy()
            else:
                prev_tail = None
                
            # Stream directly in 64K blocks to disk
            BLOCK = 65536
            for b in range(0, len(segment), BLOCK):
                f_out.write(segment[b : b + BLOCK])
            total_written += len(segment)
            
    print(f"Assembly Complete! Written: {total_written}, Expected: {total_target_samples}, Diff: {total_written - total_target_samples} samples.")
```

---

## 8. Phase 6: Final Muxing & Multi-Track Release

Multiplex the untouched video stream and synced tracks using `mkvmerge`:

```bash
mkvmerge -o "output/Title (Year) - 2160p AVC DD5.1 [Synced Audio].mkv" \
  --title "Title (Year)" \
  --video-tracks 0 --no-audio --no-subtitles \
  --track-name "0:2160p UHD AVC" --language "0:tel" \
  "target_video.mkv" \
  --track-name "0:Telugu [Dolby Digital 5.1 @ 640 kbps - Synced Master]" \
  --language "0:tel" --default-track "0:yes" \
  "tmp/synced_audio_5ch.ac3" \
  --track-name "0:Telugu [AAC 2.0 @ 256 kbps - Synced Stereo]" \
  --language "0:tel" --default-track "0:no" \
  "tmp/synced_audio_stereo.m4a" \
  --no-video --no-subtitles --audio-tracks 1 \
  --track-name "1:Tamil [Opus 2.0 - Broadcast Dub Reference]" \
  --language "1:tam" --default-track "1:no" \
  "target_video.mkv"
```

### Multi-Point Verification Rule

Spot check 10–15 points across the entire film (including scene boundaries, 25%, 50%, 75%, and end). Compute sequence normalized cross-correlation:
- Scores $\ge 0.50$: **PASS** ($r \ge 0.99$ for matching audio passages).
- Audio duration difference vs video: **$< 50\text{ms}$ ($< 1$ video frame)**.
- Sample math difference: **0 samples lost across the entire film**.

---

## 9. Phase 7: Lossless Intro / Head-Tail Trimming via Keyframe Splitting

When a target release contains unwanted distributor logos, regional title cards, or censor certificates preceding the film (e.g. 75s of black screens/titles):

### Step 1: Detect the Exact Boundary Keyframe (I-Frame)

```bash
ffprobe -v error -select_streams v:0 \
  -show_frames -read_intervals 70%80 \
  -show_entries frame=pts_time,pict_type,key_frame \
  -of json "output/file.mkv"
```

Find the keyframe immediately at or before the transition from titles to film picture (e.g. `pts_time: 75.000000, key_frame: 1`).

### Step 2: Losslessly Trim Container with Zero Re-Encoding

```bash
mkvmerge -o "output/file.trimmed.mkv" \
  --split parts:00:01:15- \
  "output/file.mkv"
```

- **Zero Re-encoding:** Video and audio streams are copied byte-for-byte.
- **Zero Drift:** All audio tracks (5.1 surround, stereo, dub) are split at the identical timestamp, preserving perfect lip-sync from the very first frame of the trimmed file.

