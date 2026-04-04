# Eye Hear U — Testing & Coverage

## Coverage targets (CI)

| Component   | Enforced metric                                                                                             | Config / command                                                                      |
| ----------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Backend** | **100%** lines **and** branches on `app/`                                                                   | `backend/.coveragerc` + `pytest --cov=app --cov-fail-under=100`                       |
| **ML**      | **100%** lines **and** branches on `i3d_msft/` and `modal_train_i3d.py`                                     | `ml/.coveragerc` + `pytest --cov=i3d_msft --cov=modal_train_i3d --cov-fail-under=100` |
| **Mobile**  | **100%** **statements**, **branches**, **lines**, and **functions** on collected `app/**` and `services/**` | `mobile/package.json` → `jest.coverageThreshold.global`                               |

**Not in scope:** `data/scripts/`, `benchmark/`, and Terraform are not measured by these gates. Add dedicated tests or CI jobs if you need them held to the same bar.

Approximate totals (re-check anytime): **backend ~156**, **ML ~194**, **mobile ~104** tests (`pytest --collect-only`, `npx jest --ci`).

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

### Coverage (XML)

```bash
cd backend
export PYTHONPATH=..
pytest tests/ --cov=app --cov-report=xml
# produces backend/coverage.xml
```

## What is tested

| Area                          | Tests                                                                                                                                                                                                                                                                                                                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lifespan / startup            | `test_main_lifespan.py` — model load OK, `FileNotFoundError`, generic `Exception`, T5 preload when `GLOSS_ENGLISH_MODE=t5`                                                                                                                                                                                                                                                    |
| Health                        | `test_health.py` — `/health`, `/ready`                                                                                                                                                                                                                                                                                                                                        |
| Predict API                   | `test_predict.py`, `test_predict_extra.py` — empty file, non-video, 503, success, `ValueError`, inference errors, empty `top_k`                                                                                                                                                                                                                                               |
| Predict sentence              | `test_predict_sentence.py` — `POST /predict/sentence` multi-clip, beam + LM, limits, 503, errors, **400** if any clip has empty top-k                                                                                                                                                                                                                                      |
| Sentence `GLOSS_ENGLISH_MODE` | `test_predict_sentence_modes.py` — `rule` / `t5` / `bedrock` (T5/Bedrock failures log WARNING and fall back to rule for the best line)                                                                                                                                                                                                                                        |
| Beam + gloss LM               | `test_beam_search.py`, `test_gloss_lm.py` — beam decode, **ValueError** if any clip has empty candidates, `GlossBeamLM` trigram/backoff, uniform fallback                                                                                                                                                                                                                     |
| Gloss → display line          | `test_gloss_to_english.py` — join + polish for `english` field                                                                                                                                                                                                                                                                                                                |
| T5 / Bedrock rewriters        | `test_gloss_to_english_t5.py`, `test_gloss_to_english_bedrock.py` — mocked inference                                                                                                                                                                                                                                                                                          |
| LM JSON builder               | `test_lm_builder.py` — label map load, sequence parsing, `build_lm_dict`                                                                                                                                                                                                                                                                                                      |
| Preprocessing                 | `test_preprocessing.py`, `test_preprocessing_coverage.py` — pad/crop helpers, cv2 branches, `preprocess_video`, ImportError path                                                                                                                                                                                                                                            |
| Preprocessing (depth)         | `test_preprocessing_depth.py` — 16 edge-case tests (10 positive, 6 negative): portrait 9:16 spatial preservation, 4K downscale, single-frame padding, [-1,1] normalization, frameskip adaptation, square aspect, center-crop geometry, interpolation selection, zero-frame error, all-reads-fail, undersized crop, missing opencv, temp file cleanup, codec crash propagation |
| Model service                 | `test_model_service.py`, `test_model_service_coverage.py` — label map formats, S3 download mock, `load_model`, `predict`, `predict_batch`, `sys.path` insert                                                                                                                                                                                                                  |
| Firebase                      | `test_firebase_service.py` — mocked `firebase_admin`                                                                                                                                                                                                                                                                                                                          |

**Total:** run `pytest tests/ --collect-only -q` from `backend/` (with `PYTHONPATH` set as below) for the current count; **100%** line and branch coverage on `app/` as configured in `.coveragerc`.

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

ML tests live in **`ml/tests/`**. Config: **`ml/pytest.ini`** (test paths), **`ml/.coveragerc`** (`fail_under = 100` on the combined report).

Coverage is measured for **`i3d_msft`** and **`modal_train_i3d.py`** only, with **branch** coverage enabled in **`ml/.coveragerc`**. A few entry points use `# pragma: no cover` (`if __name__ == "__main__"`, alternate `evaluate.py` imports).

### Run tests locally

```bash
cd ml
python -m pytest tests/ -v \
  --cov=i3d_msft --cov=modal_train_i3d \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov-fail-under=100
```

### What is tested

| Area                | Tests                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Export label map    | `test_export_label_map.py` — CSV parsing (basic, duplicates, case normalization, whitespace, empty gloss, missing column, empty CSV, sequential indices), main() CLI (basic, --inverse, parent dir creation)                                                                                                                                                                  |
| I3D backbone        | `test_pytorch_i3d.py` — Identity, MaxPool3dSamePadding (compute_pad, forward), Unit3D (batch norm, no batch norm, no activation, compute_pad), InceptionModule, InceptionI3d (all 16 early endpoints, forward, pretrained mode, no spatial squeeze, extract_features, replace_logits, remove_last)                                                                            |
| Video transforms    | `test_videotransforms.py` — RandomCrop, CenterCrop, RandomHorizontalFlip (output shape, repr, edge cases)                                                                                                                                                                                                                                                                     |
| I3D dataset         | `test_i3d_dataset.py` — load_rgb_frames (basic, empty, frameskip, upscale, truncated), video_to_tensor, ASLCitizenI3DDataset (init, custom gloss dict, missing/empty files, padding, getitem)                                                                                                                                                                                 |
| I3D training        | `test_i3d_train.py` — train_one_epoch, evaluate, build_arg_parser, \_read_split_rows, \_select_filenames_with_val_coverage, \_is_readable_video, \_write_filtered_split, \_load_compatible_checkpoint, \_upload_checkpoint_to_s3, \_set_backbone_trainable, \_build_optimizer, main() (smoke, init checkpoint, head-only epochs, epoch checkpoints, S3 upload, empty dataset) |
| I3D evaluation      | `test_i3d_evaluate.py` — get_device, \_read_split_rows, \_build_gloss_dict_from_csv, \_is_readable_video, \_write_filtered_split, \_topk_hits, \_compute_mrr_and_dcg, evaluate(), build_parser, main() (missing checkpoint, invalid topk, integration, S3 checkpoint, clip limit, custom output)                                                                              |
| I3D S3 data         | `test_i3d_s3_data.py` — get_s3_client, get_active_plan_id, download_splits, \_read_split_rows, collect_required_filenames, download_clip_subset (success, skip existing, missing key, access denied)                                                                                                                                                                          |
| Label map artifacts | `test_build_label_map_artifacts.py` — \_write_json, main() (basic, clip limit, S3 upload, clean workdir, empty filtered raises)                                                                                                                                                                                                                                               |
| Modal GPU wrapper   | `test_modal_train_i3d.py` — \_parse_run_name, \_build_train_cmd, \_build_eval_cmd, \_resolve_active_plan, \_upload_checkpoints, \_upload_run_metadata                                                                                                                                                                                                                         |

**Total:** run `pytest tests/ --collect-only -q` from `ml/` for the current count (about **194** tests as of last refresh); **100%** line and branch coverage on the scoped modules above.

## Mobile (Jest + jest-expo)

Mobile tests live in **`mobile/__tests__/`**. Jest is configured in `mobile/package.json` (`"preset": "jest-expo"`):

- **`collectCoverageFrom`** — `app/**/*.{ts,tsx}` and `services/**/*.ts` (excluding `__tests__`).
- **`coverageThreshold.global`** — **100%** on **statements**, **branches**, **lines**, and **functions**; CI fails if any metric drops.
- **`coverageReporters`** — `text`, `lcov`, and **`json-summary`** (the latter feeds the CI README / PR coverage job).

### Run tests locally

Match CI dependency install (Expo peer-deps):

```bash
cd mobile
npm ci --legacy-peer-deps   # or: npm install --legacy-peer-deps
npx jest --coverage --ci
```

### What is tested

| Area           | Tests                                                                                                                                                                                                                                                                 |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| API service    | `api.test.ts` — `isTunnelUnavailable`, `explainApiFailure`, `predictSign`, `predictSentence`, `checkHealth`, `resolveApiBaseUrl` (including `loca.lt` tunnel header)                                                                                                  |
| Home + layout  | `index.test.tsx`, `_layout.test.tsx` — home screen, navigation to camera/history, root stack                                                                                                                                                                          |
| Camera screen  | `camera.test.tsx` — see **Camera tests** below |
| History screen | `history.test.tsx` — empty state, history rendering, `timeAgo` formatting, clear history flow, AsyncStorage errors                                                                                                                                                      |

**Camera tests (`camera.test.tsx`)** cover permissions; single-sign and gallery upload; **multi-sign** (`predictSentence`, clear clips, errors); recording overlay, countdown, and pulse animation; camera facing toggle; TTS; **SignASL in-app video** (HTML fetch → `.mp4` URL list, `expo-video` `useVideoPlayer` / `VideoView` mocks, source fallback on player error, `expo-web-browser`); **platform layout** (`camera-top-controls`, `camera-mode-row`, recording overlay top insets on iOS vs Android); and edge cases (empty predicted gloss, recording with no URI, confidence threshold for auto-TTS, AsyncStorage failures, empty sentence `english`, `predictSentence` rejecting a non-`Error`, missing beam `score`, ellipsis while translating).

**Total:** **~104** tests, **100%** statements/branches/lines/functions on collected paths (see Jest summary after `npx jest --coverage --ci`).

## Continuous Integration (GitHub Actions)

Workflow: **`.github/workflows/ci.yml`**

On every **push** and **pull_request** to `main` or `master`, three test jobs run in parallel (**backend**, **ml**, **mobile**), then a fourth job **`coverage-report`** aggregates results.

### Test jobs

1. **Backend** — Python **3.11**, `pip install -r backend/requirements.txt`, `pytest` with **`--cov-fail-under=100`** (lines + branches on `app/`). Writes **`coverage-ci.json`** and uploads it as an artifact.
2. **ML** — Python **3.11**, `ml/requirements.txt` + pytest-cov, scoped **`--cov=i3d_msft --cov=modal_train_i3d`** with **lines + branches** at **100%**, **`coverage-ci.json`** artifact.
3. **Mobile** — Node **20**, `npm ci --legacy-peer-deps`, `npx jest --coverage --ci`. Jest writes **`coverage/coverage-summary.json`**, uploaded as an artifact.

### Coverage report job (`coverage-report`)

- Downloads the three JSON artifacts and runs **`.github/scripts/merge_coverage_report.py`**.
- **Pull requests (same repository only):** posts or updates a **sticky comment** (header `coverage`) via [`marocchino/sticky-pull-request-comment`](https://github.com/marocchino/sticky-pull-request-comment). Fork PRs do not receive a comment because the default `GITHUB_TOKEN` cannot comment on behalf of forks.
- **Push to `main` / `master`:** replaces only the README fragment between **`<!-- COVERAGE_TABLE_START -->`** and **`<!-- COVERAGE_TABLE_END -->`** (in the **CI and coverage** section of the root `README.md`), then commits and pushes as **`github-actions[bot]`** with **`[skip ci]`** in the message so the follow-up commit does not re-trigger CI.

Badges and table rows are generated from the JSON artifacts:

- **Backend** and **ML** Shields badges use the **lower** of line % and branch % (both are collected for these components).
- **Mobile** badge uses the **minimum** of lines, branches, statements, and functions so any Jest metric regression is visible; the table shows **lines** and **branches** only.

No third-party coverage hosting is required.

### PR comment + README maintenance

- Do not edit the auto-generated fragment between the two HTML comment markers by hand; CI overwrites it on the next **`main`** run. You **may** edit the headings and paragraphs **outside** those markers (for example the **CI and coverage** intro in **`README.md`**).
- To change badges, colors, or table layout, edit **`.github/scripts/merge_coverage_report.py`** (keep **`<!-- COVERAGE_TABLE_START -->`** and **`<!-- COVERAGE_TABLE_END -->`** in **`README.md`**).
