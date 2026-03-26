# Eye Hear U — Testing & Coverage

## Backend (pytest + pytest-cov)

All backend tests live in **`backend/tests/`**. Configuration:

- `backend/pytest.ini` — test paths, asyncio mode, `pythonpath` so `ml` imports resolve
- `backend/.coveragerc` — `source = app`, **`fail_under = 100`** (line + branch coverage on `app/`)

### Run tests locally

From the **`backend`** directory:

```bash
export PYTHONPATH=..   # repo root (contains `ml/`)
pytest tests/ -v
```

### Coverage (terminal)

```bash
cd backend
export PYTHONPATH=..
pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=100
```

### Coverage (HTML report)

```bash
cd backend
export PYTHONPATH=..
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html    # macOS
```

### Coverage (XML for Codecov / CI)

```bash
pytest tests/ --cov=app --cov-report=xml
# produces backend/coverage.xml
```

## What is tested

| Area | Tests |
|------|--------|
| Lifespan / startup | `test_main_lifespan.py` — model load OK, `FileNotFoundError`, generic `Exception` |
| Health | `test_health.py` — `/health`, `/ready` |
| Predict API | `test_predict.py`, `test_predict_extra.py` — empty file, non-video, 503, success, `ValueError`, inference errors, empty `top_k` |
| Preprocessing | `test_preprocessing.py`, `test_preprocessing_coverage.py` — pad/crop helpers, cv2 branches, `preprocess_video`, ImportError path |
| Preprocessing (depth) | `test_preprocessing_depth.py` — 16 edge-case tests (10 positive, 6 negative): portrait 9:16 spatial preservation, 4K downscale, single-frame padding, [-1,1] normalization, frameskip adaptation, square aspect, center-crop geometry, interpolation selection, zero-frame error, all-reads-fail, undersized crop, missing opencv, temp file cleanup, codec crash propagation |
| Model service | `test_model_service.py`, `test_model_service_coverage.py` — label map formats, S3 download mock, `load_model`, `predict`, `sys.path` insert |
| Firebase | `test_firebase_service.py` — mocked `firebase_admin` |

**Total:** 82 tests, **100%** line and branch coverage on `app/` as configured.

### Coverage depth: preprocessing module

`preprocessing.py` is the most complex module — the critical accuracy path between
raw phone video and the I3D model tensor. `test_preprocessing_depth.py` contains 16
targeted tests with detailed comments explaining each edge case:

**Positive tests (10):**
- Portrait 9:16 video preserves spatial detail (the original accuracy bug)
- 4K video downscaled before resize
- Single-frame video padded to 64 frames
- Normalization produces [-1, 1] range (not ImageNet)
- Full pipeline output shape and dtype
- Frameskip adapts to high-fps video (60fps)
- Square video no aspect distortion
- Center crop extracts geometrically central region
- INTER_AREA interpolation for downscaling
- INTER_LINEAR interpolation for upscaling

**Negative tests (6):**
- Zero-frame video raises ValueError
- All reads fail (truncated file) raises ValueError
- Center crop rejects undersized frames
- Missing opencv raises RuntimeError with install hint
- Temp file cleanup failure doesn't crash
- Decode errors propagate through pipeline

## ML (pytest + pytest-cov)

ML tests live in **`ml/tests/`**. Coverage configuration is in `ml/.coveragerc` (sources: `models`, `config`, `i3d_msft`, `training`, `evaluation`; excludes `__main__` guards and CUDA-only branches).

### Run tests locally

```bash
cd ml
python -m pytest tests/ -v --cov --cov-report=term-missing
```

### What is tested

| Area | Tests |
|------|--------|
| Classifier | `test_classifier.py` — forward shape, backbones (r3d_18, mc3_18, r2plus1d_18), freeze/unfreeze, dropout, unsupported backbone |
| Config | `test_config.py` — DataConfig, ModelConfig, TrainConfig defaults and custom values, composition |
| Dataset | `test_dataset.py` — init, sample collection, read_clip (pad/truncate/empty), augmentations (temporal shift, brightness, spatial crop), normalization, constants |
| Evaluation metrics | `test_evaluation.py` — top-k accuracy, precision/recall/F1, confusion matrix, seed reproducibility |
| Evaluation pipeline | `test_evaluate_extended.py` — evaluate_model (end-to-end with mock model), save_confusion_matrix_plot (matplotlib present/absent, large class count), main() orchestration |
| Export label map | `test_export_label_map.py` — CSV parsing (basic, duplicates, case normalization, whitespace, empty gloss, missing column, empty CSV, sequential indices), main() CLI (basic, --inverse, parent dir creation) |
| I3D backbone | `test_pytorch_i3d.py` — Identity, MaxPool3dSamePadding (compute_pad, forward), Unit3D (batch norm, no batch norm, no activation, compute_pad), InceptionModule, InceptionI3d (all 16 early endpoints, forward, pretrained mode, no spatial squeeze, extract_features, replace_logits, remove_last) |
| Training loop | `test_train.py` — train_one_epoch (returns loss/acc, params update, empty loader), evaluate (loss/acc, no gradients, empty loader), _worker_init_fn (seed determinism), set_seed (cudnn flags), main() (with/without label_map, early stopping, empty dataset exit) |
| Video transforms | `test_videotransforms.py` — RandomCrop, CenterCrop, RandomHorizontalFlip (output shape, repr, edge cases) |

**Total:** 144 tests, **100%** line coverage.

## Mobile (Jest + jest-expo)

Mobile tests live in **`mobile/__tests__/`**. Configuration is in `mobile/package.json` (`"preset": "jest-expo"`).

### Run tests locally

```bash
cd mobile
npx jest --coverage
```

### What is tested

| Area | Tests |
|------|--------|
| API service | `api.test.ts` — `isTunnelUnavailable`, `explainApiFailure`, `predictSign`, `checkHealth`, `resolveApiBaseUrl` branches |
| Camera screen | `camera.test.tsx` — permissions, recording flow, upload flow, camera toggle, error handling, prediction display, TTS |
| History screen | `history.test.tsx` — empty state, history rendering, `timeAgo` formatting, clear history flow, AsyncStorage errors |

**Total:** 59 tests, **100%** line and function coverage.

## Continuous Integration (GitHub Actions)

Workflow: **`.github/workflows/ci.yml`**

On every **push** and **pull_request** to `main` or `master`, three jobs run in parallel:

### Backend job
1. Sets up Python **3.11**
2. `pip install -r backend/requirements.txt`
3. Runs `pytest` — fails if coverage **< 100%**
4. Uploads `backend/coverage.xml` to Codecov

### ML job
1. Sets up Python **3.11**
2. `pip install -r ml/requirements.txt` + pytest-cov
3. Runs `pytest` with `.coveragerc` config — fails if coverage **< 100%**
4. Uploads `ml/coverage.xml` to Codecov

### Mobile job
1. Sets up Node.js **20**
2. `npm ci`
3. Runs `npx jest --coverage --ci`

### Codecov setup (README badge + PR comments)

1. Sign in at [codecov.io](https://about.codecov.io/) with GitHub.
2. Enable the **EyeHearU** repository.
3. In GitHub: **Settings → Secrets and variables → Actions** → add **`CODECOV_TOKEN`**.
4. Open a PR — Codecov comments with diff coverage after the first upload.

The README badge URL:

`https://codecov.io/gh/MariaMa-GitHub/EyeHearU/branch/main/graph/badge.svg`

…updates once Codecov has processed `main`.

### Without Codecov

CI still passes or fails on **`--cov-fail-under=100`** (backend). The external badge and PR bot are omitted until `CODECOV_TOKEN` is configured.
