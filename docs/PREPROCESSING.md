# Eye Hear U — Inference preprocessing (I3D)

This document describes **`backend/app/services/preprocessing.py`** and how it aligns with the **I3D training dataloader** (same frame sampling, normalization, and compatible spatial resize).

## Goal

Turn an uploaded **video file (bytes)** into a single PyTorch tensor:

**Shape:** `(1, 3, 64, 224, 224)`
**Layout:** batch **N=1**, channels **C=3**, time **T=64**, height **H=224**, width **W=224**
**Values:** approximately in **`[-1, 1]`** per channel (same scaling as training)

The **Inception I3D** model expects a **spatiotemporal** volume, not a single image.

## ASL Citizen (training) vs. mobile recordings

| | Training clips | App recordings (iOS / Expo) |
|---|----------------|------------------------------|
| **Resolution** | Normalized dataset sizes | **720p-4K**, variable by device & camera |
| **Aspect** | Often near **square / hands** in frame | **Portrait 9:16**, landscape, front cam |
| **Duration** | Dataset segment length | **~3 s** at **30-60 fps** |
| **Codec** | Pipeline output | **H.264 / HEVC** from Camera |

The pipeline uses a **short-side-256** resize (scale so the shorter dimension equals 256, preserving aspect ratio) followed by center-crop 224x224. This produces frames at the same spatial scale as training (~256-scale frames center-cropped to 224) while correctly handling **all aspect ratios**, including portrait phone video.

**Previous approach (replaced):** The original min-226 / max-256 sequential resize collapsed portrait (9:16) video width to ~144 px before re-enlarging, destroying spatial detail. The short-side-256 approach eliminates this bottleneck.

## Pipeline steps (in order)

### 1. Persist bytes to a temporary file

OpenCV's `VideoCapture` reads from a **file path**, not raw memory. The implementation writes `video_bytes` to a **NamedTemporaryFile** with suffix `.mp4`, then deletes it in a `finally` block (errors when deleting are ignored so cleanup never masks real failures).

### 2. Decode frames with `_load_rgb_frames`

This function mirrors the temporal logic of the training helper **`load_rgb_frames_from_video`** from the I3D training codebase.

#### 2a. Frame count and adaptive **frame skip**

- Read `CAP_PROP_FRAME_COUNT`. If **<= 0**, treat as unreadable -> return **empty** array (later: "no decodable frames").
- **frameskip** depends on length:
  - **< 96** frames -> `frameskip = 1`
  - **>= 96** -> `frameskip = 2`
  - **>= 160** -> `frameskip = 3`

**Rationale:** Longer clips subsample in time so not every frame is read (speed + closer to ASL-Citizen-style windowing). Shorter phone clips stay dense (`frameskip = 1`) for temporal context.

#### 2b. Temporal **window** (center bias)

Start index `start` is chosen so decoding is **centered** in the clip, with different offsets when `frameskip` is 2 or 3 (same formulas as training). **Rationale:** Reduces bias from idle frames at the start/end of a recording.

#### 2c. Spatial resize (per frame, BGR `uint8`)

1. **Mobile coarse cap (`MOBILE_MAX_LONG_SIDE = 1280`)**
   If `max(H, W) > 1280`, downscale uniformly so the long side is **1280** using **INTER_AREA**.
   **Rationale:** Very large frames (e.g. **4K**) are costly for OpenCV and memory; the model only needs context for a **224x224** crop.

2. **Short-side-256 resize (`RESIZE_SIDE = 256`)**
   Scale so `min(H, W) = 256`, preserving aspect ratio. Uses **INTER_AREA** when shrinking, **INTER_LINEAR** when enlarging.

   **Rationale:** The model was trained on ~256-scale frames center-cropped to 224. This single-step resize produces the same spatial scale for **all** aspect ratios (portrait, landscape, square) without the spatial destruction that the old min-226/max-256 sequential logic caused on elongated frames.

   **Example (portrait 1080x1920):**
   - Old: 720x1280 -> 144x256 (width crushed!) -> upscale 225x399 -> crop 224x224
   - New: 720x1280 -> 256x456 (short side = 256) -> crop 224x224

#### 2d. Color and normalization

- OpenCV reads **BGR** -> convert to **RGB**.
- Normalize: **`pixel / 255 * 2 - 1`** -> **`[-1, 1]`**.

**Rationale:** I3D training in this repo uses this range, **not** ImageNet mean/std. ImageNet normalization would **shift the input distribution** and **hurt accuracy**.

#### 2e. Read loop

Read up to `limit` raw steps from the capture, keeping every `frameskip`-th frame. Release the capture when done.

If no frames were collected, return an **empty** array (shape `(0, ...)`).

### 3. Pad or trim to **64** frames (`_pad_frames`)

- If **more than 64** frames -> keep the **first 64** (same as training's upper bound for `total_frames`).
- If **fewer** -> **tile the last frame** until length is 64.

**Rationale:** The model expects a **fixed temporal length**. Short clips are padded with the **last** frame to avoid introducing fake motion from the **first** frame (which might be pre-sign idle).

### 4. **Center crop** to 224x224 (`_center_crop`)

For each of the 64 frames, crop a **224x224** window centered in **HxW**. Since the short-side-256 resize guarantees `min(H, W) = 256 >= 224`, the crop is always valid.

**Rationale:** Training uses **RandomCrop(224)** on train and **CenterCrop(224)** on val/test. At inference only **center** crop is used -- no randomness, reproducible, aligned with **eval**.

### 5. Tensor layout

- Numpy array shape `(T, H, W, C)` -> transpose to **`(C, T, H, W)`** -> add batch dimension **`(1, C, T, H, W)`**.

**Rationale:** PyTorch 3D convs in this codebase use **channels-first** and **time after channels**.

## Failure modes

| Situation | Behavior |
|-----------|----------|
| OpenCV missing | `RuntimeError` asking for `opencv-python-headless` |
| No frames decoded | `ValueError("Video has no decodable frames")` |
| Temp file delete fails | `OSError` swallowed in `finally` (file may remain in `/tmp` on some OSes -- rare) |

## Intentionally omitted

- **No ImageNet mean/std** -- incorrect for this I3D training recipe.
- **No 16-frame R3D-style sampling** -- this model is **I3D @ 64** frames.
- **No random augmentations at inference** -- would make demos non-reproducible.

## Keeping training and inference in sync

Whenever training code changes in the I3D **`dataset.py`** / **`videotransforms.py`** on the training branch, update **`preprocessing.py`** and re-run:

```bash
cd backend
export PYTHONPATH=..
pytest tests/ --cov=app --cov-fail-under=100
```

Add a regression test when numerics change (e.g. golden tensor stats on a fixture video).
