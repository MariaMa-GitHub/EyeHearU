# Inference preprocessing (I3D) — design and rationale

This document describes **`backend/app/services/preprocessing.py`** and how it aligns with the **I3D training dataloader** on the training branch (same frame sampling, resize rules, and `[-1, 1]` normalization).

## Goal

Turn an uploaded **video file (bytes)** into a single PyTorch tensor:

**Shape:** `(1, 3, 64, 224, 224)`  
**Layout:** batch **N=1**, channels **C=3**, time **T=64**, height **H=224**, width **W=224**  
**Values:** approximately in **`[-1, 1]`** per channel (same scaling as training)

The **Inception I3D** model expects a **spatiotemporal** volume, not a single image.

## ASL Citizen (training) vs. mobile recordings

| | Training clips | App recordings (iOS / Expo) |
|---|----------------|------------------------------|
| **Resolution** | Normalized dataset sizes | **720p–4K**, variable by device & camera |
| **Aspect** | Often near **square / hands** in frame | **Portrait 9:16**, landscape, front cam |
| **Duration** | Dataset segment length | **~3 s** at **30–60 fps** |
| **Codec** | Pipeline output | **H.264 / HEVC** from Camera |

The pipeline keeps the **same numeric recipe** as the I3D dataloader (226 / 256 / 64 / 224 / `[-1,1]`) so weights remain valid, and adds **mobile-specific** steps: **coarse downscale** if the long side exceeds **1280 px** (4K is unnecessary for a 224×224 classifier), **INTER_AREA** when shrinking raw BGR frames, and **guarantee both sides ≥ 224** before center-crop so panoramic strips do not break I3D.

## Pipeline steps (in order)

### 1. Persist bytes to a temporary file

OpenCV’s `VideoCapture` reads from a **file path**, not raw memory. The implementation writes `video_bytes` to a **NamedTemporaryFile** with suffix `.mp4`, then deletes it in a `finally` block (errors when deleting are ignored so cleanup never masks real failures).

### 2. Decode frames with `_load_rgb_frames`

This function mirrors the training helper **`load_rgb_frames_from_video`** from the I3D training codebase (same logic as in `dataset.py` on the aligned training branch).

#### 2a. Frame count and adaptive **frame skip**

- Read `CAP_PROP_FRAME_COUNT`. If **≤ 0**, treat as unreadable → return **empty** array (later: “no decodable frames”).
- **frameskip** depends on length:
  - **&lt; 96** frames → `frameskip = 1`
  - **≥ 96** → `frameskip = 2`
  - **≥ 160** → `frameskip = 3`

**Rationale:** Longer clips subsample in time so not every frame is read (speed + closer to MS/ASL-Citizen-style windowing). Shorter phone clips stay dense (`frameskip = 1`) for temporal context.

#### 2b. Temporal **window** (center bias)

Start index `start` is chosen so decoding is **centered** in the clip, with different offsets when `frameskip` is 2 or 3 (same formulas as training). **Rationale:** Reduces bias from idle frames at the start/end of a recording.

#### 2c. Spatial resize rules (per frame, BGR `uint8`)

1. **Mobile coarse cap (`MOBILE_MAX_LONG_SIDE = 1280`)**  
   If `max(H, W) > 1280`, downscale uniformly so the long side is **1280** using **INTER_AREA**.  
   **Rationale:** Very large frames (e.g. **4K**) are costly for OpenCV and memory; the model only needs context for a **224×224** crop.

2. **Training-style rules** (same as `load_rgb_frames_from_video`):  
   - If **min(H, W) &lt; 226**, scale up (short side **226**) with **INTER_LINEAR**.  
   - If **max(H, W) &gt; 256**, scale down (long side **≤ 256**) with **INTER_AREA**.

**Rationale:** Matches the training resolution pipeline before cropping. **Phones** can still hit edge cases (e.g. **256×51** after the max-256 step) because **portrait** aspect ratios differ from many **ASL Citizen** crops — see step 4.

#### 2d. Color and normalization

- OpenCV reads **BGR** → convert to **RGB**.
- Normalize: **`pixel / 255 * 2 - 1`** → **`[-1, 1]`**.

**Rationale:** I3D training in this repo uses this range, **not** ImageNet mean/std. ImageNet normalization would **shift the input distribution** and **hurt accuracy**.

#### 2e. Read loop

Read up to `limit` raw steps from the capture, keeping every `frameskip`-th frame. Release the capture when done.

If no frames were collected, return an **empty** array (shape `(0, …)`).

### 3. Pad or trim to **64** frames (`_pad_frames`)

- If **more than 64** frames → keep the **first 64** (same as training’s upper bound for `total_frames`).
- If **fewer** → **tile the last frame** until length is 64.

**Rationale:** The model expects a **fixed temporal length**. Short clips are padded with the **last** frame to avoid introducing fake motion from the **first** frame (which might be pre-sign idle).

### 4. **Ensure both sides ≥ 224** (`_ensure_both_sides_at_least`)

After the training-style min-side / max-side (226 / 256) logic, **very wide or tall** frames can still end up with one dimension **under 224** (e.g. **256×51** after scaling). A naive **center crop** then does not produce a true 224×224 volume and I3D can fail internally (`kernel size` larger than feature map).

The code **upscales** so **min(H, W) ≥ 224** while preserving aspect ratio, using the same float `[-1, 1]` frames and linear interpolation.

**Rationale:** Phone recordings vary in aspect ratio more than curated dataset clips; this step is an **inference safety** net that preserves the intent of the pipeline (still center-crop 224 like eval).

### 5. **Center crop** to 224×224 (`_center_crop`)

For each of the 64 frames, crop a **224×224** window centered in **H×W**. If dimensions are still too small (should not happen after step 4), a clear `ValueError` is raised.

**Rationale:** Training uses **RandomCrop(224)** on train and **CenterCrop(224)** on val/test. At inference only **center** crop is used — no randomness, reproducible, aligned with **eval**.

### 6. Tensor layout

- Numpy array shape `(T, H, W, C)` → transpose to **`(C, T, H, W)`** → add batch dimension **`(1, C, T, H, W)`**.

**Rationale:** PyTorch 3D convs in this codebase use **channels-first** and **time after channels**.

## Failure modes

| Situation | Behavior |
|-----------|----------|
| OpenCV missing | `RuntimeError` asking for `opencv-python-headless` |
| No frames decoded | `ValueError("Video has no decodable frames")` |
| Temp file delete fails | `OSError` swallowed in `finally` (file may remain in `/tmp` on some OSes — rare) |

## Intentionally omitted

- **No ImageNet μ/σ** — incorrect for this I3D training recipe.  
- **No 16-frame R3D-style sampling** — this model is **I3D @ 64** frames.  
- **No random augmentations at inference** — would make demos non-reproducible.

## Keeping training and inference in sync

Whenever training code changes in the I3D **`dataset.py`** / **`videotransforms.py`** on the training branch, update **`preprocessing.py`** and re-run:

```bash
cd backend
export PYTHONPATH=..
pytest tests/ --cov=app --cov-fail-under=100
```

Add a regression test when numerics change (e.g. golden tensor stats on a fixture video).
