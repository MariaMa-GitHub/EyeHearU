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
| Predict sentence | `test_predict_sentence.py` — `POST /predict/sentence` multi-clip, beam + LM, limits, 503, errors |
| Beam + gloss LM | `test_beam_search.py`, `test_gloss_lm.py` — beam decode, `GlossBeamLM` trigram/backoff, uniform fallback |
| Gloss → display line | `test_gloss_to_english.py` — join + polish for `english` field |
| LM JSON builder | `test_lm_builder.py` — label map load, sequence parsing, `build_lm_dict` |
| Preprocessing | `test_preprocessing.py`, `test_preprocessing_coverage.py` — pad/crop helpers, cv2 branches, `preprocess_video`, ImportError path |
| Preprocessing (depth) | `test_preprocessing_depth.py` — 16 edge-case tests (10 positive, 6 negative): portrait 9:16 spatial preservation, 4K downscale, single-frame padding, [-1,1] normalization, frameskip adaptation, square aspect, center-crop geometry, interpolation selection, zero-frame error, all-reads-fail, undersized crop, missing opencv, temp file cleanup, codec crash propagation |
| Model service | `test_model_service.py`, `test_model_service_coverage.py` — label map formats, S3 download mock, `load_model`, `predict`, `predict_batch`, `sys.path` insert |
| Firebase | `test_firebase_service.py` — mocked `firebase_admin` |

**Also:** `test_gloss_to_english.py` (offline gloss line formatter), `test_lm_builder.py` (offline `gloss_lm.json` builder).

**Total:** 124+ tests, **100%** line and branch coverage on `app/` as configured (run `pytest --collect-only` for the current count).

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

ML tests live in **`ml/tests/`**. Coverage configuration is in `ml/.coveragerc` (sources: `i3d_msft`, `modal_train_i3d`; excludes `__main__` guards and CUDA-only branches).

### Run tests locally

```bash
cd ml
python -m pytest tests/ -v --cov --cov-report=term-missing
```

### What is tested

| Area | Tests |
|------|--------|
| Export label map | `test_export_label_map.py` — CSV parsing (basic, duplicates, case normalization, whitespace, empty gloss, missing column, empty CSV, sequential indices), main() CLI (basic, --inverse, parent dir creation) |
| I3D backbone | `test_pytorch_i3d.py` — Identity, MaxPool3dSamePadding (compute_pad, forward), Unit3D (batch norm, no batch norm, no activation, compute_pad), InceptionModule, InceptionI3d (all 16 early endpoints, forward, pretrained mode, no spatial squeeze, extract_features, replace_logits, remove_last) |
| Video transforms | `test_videotransforms.py` — RandomCrop, CenterCrop, RandomHorizontalFlip (output shape, repr, edge cases) |
| I3D dataset | `test_i3d_dataset.py` — load_rgb_frames (basic, empty, frameskip, upscale, truncated), video_to_tensor, ASLCitizenI3DDataset (init, custom gloss dict, missing/empty files, padding, getitem) |
| I3D training | `test_i3d_train.py` — train_one_epoch, evaluate, build_arg_parser, _read_split_rows, _select_filenames_with_val_coverage, _is_readable_video, _write_filtered_split, _load_compatible_checkpoint, _upload_checkpoint_to_s3, _set_backbone_trainable, _build_optimizer, main() (smoke, init checkpoint, head-only epochs, epoch checkpoints, S3 upload, empty dataset) |
| I3D evaluation | `test_i3d_evaluate.py` — get_device, _read_split_rows, _build_gloss_dict_from_csv, _is_readable_video, _write_filtered_split, _topk_hits, _compute_mrr_and_dcg, evaluate(), build_parser, main() (missing checkpoint, invalid topk, integration, S3 checkpoint, clip limit, custom output) |
| I3D S3 data | `test_i3d_s3_data.py` — get_s3_client, get_active_plan_id, download_splits, _read_split_rows, collect_required_filenames, download_clip_subset (success, skip existing, missing key, access denied) |
| Label map artifacts | `test_build_label_map_artifacts.py` — _write_json, main() (basic, clip limit, S3 upload, clean workdir, empty filtered raises) |
| Modal GPU wrapper | `test_modal_train_i3d.py` — _parse_run_name, _build_train_cmd, _build_eval_cmd, _resolve_active_plan, _upload_checkpoints, _upload_run_metadata |

**Total:** 190+ tests, **100%** line coverage.

## Mobile (Jest + jest-expo)

Mobile tests live in **`mobile/__tests__/`**. Jest is configured in `mobile/package.json` (`"preset": "jest-expo"`):

- **`collectCoverageFrom`** — `app/**` and `services/**` only (TypeScript sources under test).
- **`coverageThreshold`** — **100%** lines and **100%** functions globally on those paths; CI fails if coverage drops.

### Run tests locally

```bash
cd mobile
npx jest --coverage
```

### What is tested

| Area | Tests |
|------|--------|
| API service | `api.test.ts` — `isTunnelUnavailable`, `explainApiFailure`, `predictSign`, **`predictSentence`**, `checkHealth`, `resolveApiBaseUrl` (including `loca.lt` tunnel header) |
| Home + layout | `index.test.tsx`, `_layout.test.tsx` — home screen, navigation to camera/history, root stack |
| Camera screen | `camera.test.tsx` — permissions, single-sign flow, **multi-sign** (mode switch, `predictSentence`, clear, errors), countdown, upload, toggle, TTS |
| History screen | `history.test.tsx` — empty state, history rendering, `timeAgo` formatting, clear history flow, AsyncStorage errors |

**Total:** 77+ tests, **100%** line and function coverage on `app/` and `services/` (run `npx jest --listTests` / full Jest output for the current count).

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
3. Runs `npx jest --coverage --ci` — fails if coverage falls below **100%** lines or **100%** functions on `app/` and `services/` (see `coverageThreshold` in `mobile/package.json`)
4. Uploads `mobile/coverage/lcov.info` to Codecov (flag: **`mobile`**) so the dashboard includes frontend with backend and ML

`codecov.yml` defines flags for **`backend`**, **`ml`**, and **`mobile`**, plus **`fixes`** so Jest’s `SF:app/...` and `SF:services/...` paths resolve under `mobile/` in the UI.

### Codecov setup (README badge + PR comments)

1. Sign in at [codecov.io](https://about.codecov.io/) with GitHub.
2. Enable the **EyeHearU** repository.
3. In GitHub: **Settings → Secrets and variables → Actions** → add **`CODECOV_TOKEN`**.
4. Open a PR — after all three jobs upload, Codecov merges reports and comments with per-flag coverage (wait is configured via `codecov.notify.after_n_builds: 3`).

The README badge URL:

`https://codecov.io/gh/MariaMa-GitHub/EyeHearU/branch/main/graph/badge.svg`

…updates once Codecov has processed `main`.

### Without Codecov

CI still passes or fails on **`--cov-fail-under=100`** (backend and ML) and Jest **`coverageThreshold`** (mobile). The external badge and PR bot are omitted until `CODECOV_TOKEN` is configured.
