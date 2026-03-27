# Assignment 5 Writeup — Eye Hear U

## Part One: Profiling Execution Time

We profiled **5 key functions** using Python's [`cProfile`](https://docs.python.org/3/library/profile.html) module. The profiling script lives at [`ml/profiling/profile_functions.py`](../ml/profiling/profile_functions.py) and generates per-function cProfile tables, binary `.prof` files for pstats/snakeviz analysis, and a machine-readable JSON summary.

### How to run

```bash
cd ml/
python -m profiling.profile_functions
```

### Profiled functions

| # | Function | Location | Wall Time | Bottleneck | Optimization |
|---|----------|----------|-----------|------------|-------------|
| 1 | `preprocess_video` | `backend/app/services/preprocessing.py` | 0.93 s (3 calls) | `cv2.resize` (77%) | Combine the two-step resize (mobile cap + short-side-256) into a single scale factor, halving resize calls (~35% savings) |
| 2 | `predict` | `backend/app/services/model_service.py` | 4.11 s (3 calls) | `torch.conv3d` (60%) + `max_pool3d` (30%) | JIT trace / `torch.compile`, ONNX Runtime export, FP16 inference, or reduced input resolution (160x160) |
| 3 | `i3d_evaluate` | `ml/i3d_msft/evaluate.py` | 6.17 s (20 samples) | `torch.conv3d` (96%) | Same inference optimizations as `predict` + larger batch size to amortize overhead |
| 4 | `i3d_train_one_epoch` | `ml/i3d_msft/train.py` | 20.53 s (1 epoch) | `run_backward` (66%) + `conv3d` (30%) | Mixed-precision AMP (`torch.cuda.amp`), freeze early backbone layers, `torch.compile` |
| 5 | `build_gloss_dict_from_csv` | `ml/i3d_msft/export_label_map.py` | 0.75 s (50 calls x 5K rows) | `csv.DictReader.__next__` (56%) | Switch to `csv.reader` with column index (avoids per-row dict construction), cache with `lru_cache` |

### Key takeaways

1. **3D convolutions dominate** — `torch.conv3d` accounts for 60-96% of time in all model-related functions. This is inherent to the I3D architecture.
2. **Backpropagation is 2x the forward pass** — gradient computation through 3D conv layers is the single largest cost in training.
3. **Preprocessing is resize-bound** — merging the two-step resize into one operation would cut `preprocess_video` time by ~35%.
4. **CSV parsing overhead is avoidable** — `DictReader` creates unnecessary per-row dictionaries; `csv.reader` eliminates this.
5. **Post-processing is negligible** — metric computation takes <1% of time in evaluation.

Full detailed analysis: [`docs/PROFILING.md`](PROFILING.md)

---

## Part Two: Code Coverage

### Coverage tools

- **Backend + ML:** [`pytest-cov`](https://github.com/pytest-dev/pytest-cov) (pytest plugin wrapping `coverage.py`)
- **Mobile:** Jest with built-in coverage (`--coverage` flag)

### GitHub Actions workflow

The CI workflow ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs three parallel jobs on every push/PR to `main`:

| Job | Command | Enforcement | Coverage Upload |
|-----|---------|-------------|-----------------|
| Backend | `pytest --cov=app --cov-report=xml --cov-fail-under=100` | 100% line + branch | Codecov (flag: `backend`) |
| ML | `pytest --cov --cov-report=xml --cov-fail-under=100` | 100% line | Codecov (flag: `ml`) |
| Mobile | `npx jest --coverage --ci` | 100% function (Jest thresholds) | — |

### README badge

The README includes a live Codecov badge that updates automatically when CI uploads coverage reports:

```
[![codecov](https://codecov.io/gh/MariaMa-GitHub/EyeHearU/branch/main/graph/badge.svg)](https://codecov.io/gh/MariaMa-GitHub/EyeHearU)
```

### Codecov PR integration

The [`codecov.yml`](../codecov.yml) configures:
- **Project status:** auto-target from main baseline, 1% regression threshold
- **Patch status:** new code must be 80% covered
- **Flags:** separate tracking for `backend` and `ml` with carryforward
- **PR comments:** diff coverage, per-flag breakdown, file-level detail

### How to generate coverage locally

```bash
# Backend (100% line + branch)
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=100

# ML (100% line)
cd ml
python -m pytest tests/ -v --cov --cov-report=term-missing --cov-fail-under=100

# Mobile (100% function)
cd mobile
npx jest --coverage
```

---

## Part Three: Test Coverage Breadth

### Current coverage

| Component | Tests | Line Coverage | Branch Coverage |
|-----------|-------|--------------|-----------------|
| Backend (`app/`) | 82 pytest | **100%** | **100%** |
| ML (`i3d_msft/`, `modal_train_i3d`) | 191 pytest | **100%** | — |
| Mobile | 59 Jest | **100%** line | **100%** function |
| **Total** | **332 tests** | **100%** | |

Coverage is enforced in CI with `--cov-fail-under=100` for backend and ML. Any PR that drops below 100% will fail the CI check.

### What is covered

**Backend (82 tests):**
- API endpoints: `/health`, `/ready`, `POST /api/v1/predict`
- Video preprocessing: frame decode, resize, crop, normalize, pad, temporal sampling
- Model service: S3 download, label map loading, inference, top-k predictions
- Lifespan: model load success/failure, startup error handling
- Firebase: mocked Firestore integration

**ML (191 tests):**
- I3D backbone: all Inception modules, forward pass, weight loading, logits replacement
- I3D dataset: frame loading, frameskip, normalization, padding, tensor conversion
- I3D training: train_one_epoch, evaluate, checkpoint handling, S3 upload, optimizer building
- I3D evaluation: top-k hits, MRR, DCG, split parsing, device detection
- S3 data: client setup, split downloads, clip subset management
- Export/build label map: CSV parsing, JSON writing, S3 upload
- Video transforms: random crop, center crop, horizontal flip
- Modal wrapper: run name parsing, command building, checkpoint resolution

**Mobile (59 tests):**
- API service: health check, predict, URL resolution, error explanations
- Camera screen: permissions, recording, upload, camera toggle, prediction display, TTS
- History screen: empty state, rendering, time formatting, clear history, storage errors

---

## Part Three: Test Coverage Depth

### Chosen module: `backend/app/services/preprocessing.py`

**Why this module?** `preprocessing.py` is the critical accuracy path between a raw phone video and the I3D model input tensor. Every frame passes through temporal sampling, spatial resize, normalization, padding, and center-cropping. A bug in any stage silently degrades prediction accuracy — the root cause of our original low-accuracy bug was a spatial resize error here. The module must handle arbitrary aspect ratios (portrait 9:16, landscape 16:9, square, ultra-wide), extreme resolutions (4K, tiny webcam), variable-length videos, corrupt inputs, and missing dependencies.

### Test file: [`backend/tests/test_preprocessing_depth.py`](../backend/tests/test_preprocessing_depth.py)

The file contains **16 targeted tests** (10 positive, 6 negative), each with detailed comments explaining the edge case, why it matters, and what it validates.

#### Positive tests (correct behavior)

| # | Test | Edge case / Use case | Why it matters |
|---|------|---------------------|---------------|
| 1 | `test_portrait_9_16_preserves_spatial_detail` | Portrait phone video (1080x1920) | **The original accuracy bug** — old pipeline crushed portrait width to 144px |
| 2 | `test_4k_video_downscaled_before_resize` | 4K video (3840x2160) | Without mobile cap, 4K frames waste memory and amplify codec artifacts |
| 3 | `test_single_frame_video_padded_to_64` | Video with only 1 frame | Very short recordings must pad to 64 frames, not crash |
| 4 | `test_normalization_range_minus1_to_plus1` | [-1,1] normalization | I3D uses [-1,1], NOT ImageNet mean/std — wrong normalization silently degrades accuracy |
| 5 | `test_full_pipeline_output_shape_and_dtype` | End-to-end pipeline | Model expects exactly (1, 3, 64, 224, 224) float32 — any deviation crashes or mismatches |
| 6 | `test_frameskip_adapts_to_high_fps_video` | 200-frame 60fps video | Without adaptive frameskip, 60fps video oversamples the beginning and misses the end |
| 7 | `test_square_video_no_aspect_distortion` | Square 1:1 video | Short-side-256 must work when both sides are equal |
| 8 | `test_center_crop_extracts_center_region` | Crop geometry | ASL signs are centered in frame — off-center crop cuts off hands/fingers |
| 9 | `test_resize_uses_area_interpolation_when_shrinking` | Downscale interpolation | INTER_AREA avoids aliasing; INTER_LINEAR creates artifacts when shrinking |
| 10 | `test_resize_uses_linear_interpolation_when_enlarging` | Upscale interpolation | INTER_AREA is undefined for upscaling; INTER_LINEAR produces smooth results |

#### Negative tests (error handling)

| # | Test | Failure mode | Why it matters |
|---|------|-------------|---------------|
| 11 | `test_zero_frame_video_raises_value_error` | Corrupt container, 0 frames | Must raise ValueError, not pass empty tensor to model |
| 12 | `test_all_reads_fail_raises_value_error` | Valid header, corrupt frame data | Truncated files must not produce garbage predictions |
| 13 | `test_center_crop_rejects_undersized_frames` | Frames smaller than crop size | Defensive check against upstream resize failure |
| 14 | `test_missing_opencv_raises_runtime_error` | cv2 not installed | Must tell user what to install, not show cryptic ImportError |
| 15 | `test_temp_file_cleanup_failure_does_not_crash` | os.unlink fails (permissions) | OSError from cleanup must not propagate — prediction is still valid |
| 16 | `test_decode_error_propagates_through_pipeline` | Unexpected codec crash | Unexpected errors must propagate to caller, not be silently swallowed |

---

## Part Four: Two Memorable Bugs

TODO — video walkthrough to be recorded and linked here.
