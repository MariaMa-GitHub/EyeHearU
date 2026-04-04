# Eye Hear U

[![Coverage](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml)

**Record American Sign Language (ASL) on your phone → get English text on screen and optional spoken output.**  
Monorepo: Expo (React Native) mobile app, FastAPI inference API, PyTorch I3D model code, plus optional data pipeline, cloud training (Modal), and AWS (Terraform).

---

## About this project

**Eye Hear U** is a **monorepo** built around an **Expo** mobile client and a **FastAPI** backend: you record short **ASL video clips**, send them to the API, and get **gloss labels** (short English-like words the model was trained on) plus a **readable line of text** and optional **text-to-speech** on the device. It targets **practice and quick communication support**, not certified interpreting.

| If you are… | Start here |
|-------------|------------|
| New to the repo | [Getting started](#getting-started) |
| Using the app | [User guide](docs/USER_GUIDE.md) |
| Developing or debugging | [Developer guide](docs/DEVELOPER_GUIDE.md) |
| Changing how translation works | [ASL translation pipeline](docs/ASL_TRANSLATION_PIPELINE.md) |

**Modes:** **Single sign** — one clip, one prediction (plus alternates). **Multi-sign** — several clips in order; the server runs **batched I3D**, **beam search**, and a **gloss n-gram language model**, then formats an **English** line.

---

## Features

- Mobile: camera and gallery upload, single/multi-sign flows, TTS, SignASL-style reference playback, **on-device history** (`AsyncStorage`).
- API: `POST /api/v1/predict`, `POST /api/v1/predict/sentence`, health endpoints; loads **I3D** + **`gloss_lm.json`** at startup; can pull **weights from S3** if not cached locally.
- ML: **Inception I3D** training/eval in `ml/i3d_msft/`, **Modal** wrapper for GPU training, label map JSON in repo.
- Ops: **Docker** / **docker-compose**, **Terraform** modules, **Kubernetes** manifests under `infrastructure/k8s/`.

**Not wired end-to-end:** `backend/app/services/firebase_service.py` is an **optional** Firestore helper; the shipped app does **not** call it from startup, and the mobile client does **not** sync history to the cloud.

---

## Getting started

You will run **two processes**: the **API** (Python) and the **mobile app** (Node/Expo). A **phone on the same Wi‑Fi** needs your computer’s **LAN IP**, not `localhost`.

### Prerequisites

| Tool | Notes |
|------|--------|
| **Python 3.11+** | Matches CI; use `python3` if `python` is missing |
| **Node.js 20+** | Matches CI; use `npm ci --legacy-peer-deps` in `mobile/` |
| **Git** | Clone this repository |

Optional: **AWS credentials** if the API should download weights from S3 automatically; otherwise place `best_model.pt` under `backend/model_cache/` (see [Developer guide](docs/DEVELOPER_GUIDE.md)).

### Run the API

From the **repository root**, `PYTHONPATH` must include the root so `import ml` works.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # adjust MODEL_DEVICE, paths, optional Bedrock/T5
export PYTHONPATH=..               # parent directory = repo root (required)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check: `curl http://localhost:8000/health`

### Run the mobile app

In a **second terminal**:

```bash
cd mobile
npm install --legacy-peer-deps     # or: npm ci --legacy-peer-deps
cp .env.example .env
```

Edit **`mobile/.env`** so the device can reach the API:

```bash
# Replace with your machine's LAN IP when using a physical phone
EXPO_PUBLIC_API_URL=http://192.168.1.50:8000
```

Start Expo (LAN is usually easiest when phone and PC share Wi‑Fi):

```bash
npm run start:lan
# or: npx expo start
```

Scan the QR code with **Expo Go**, or press `i` for the iOS Simulator. For the simulator on the same Mac, `http://127.0.0.1:8000` is often enough.

**Common pitfall:** Metro’s URL (port **8081**) is **not** the API URL (port **8000**). See [Developer guide](docs/DEVELOPER_GUIDE.md) for tunnels and iOS Local Network permission.

### Try the API with curl (optional)

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict -F "file=@/path/to/clip.mp4"
curl -X POST "http://localhost:8000/api/v1/predict/sentence?beam_size=8&lm_weight=1" \
  -F "files=@/path/to/a.mp4" -F "files=@/path/to/b.mp4"
```

---

## Repository structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, lifespan (model + LM)
│   │   ├── config.py            # Pydantic settings / env
│   │   ├── routers/             # health, predict
│   │   ├── schemas/             # prediction.py
│   │   └── services/            # model, preprocessing, beam, LM, gloss→English, …
│   ├── data/gloss_lm.json
│   ├── scripts/build_gloss_lm.py
│   ├── tests/
│   └── requirements.txt
├── mobile/
│   ├── app/                     # Expo Router screens
│   ├── services/api.ts
│   ├── __tests__/
│   └── package.json
├── ml/
│   ├── i3d_msft/                # I3D code, train, evaluate, dataset, S3 helpers
│   ├── i3d_label_map_mvp-sft-full-v1.json
│   ├── modal_train_i3d.py
│   ├── tests/
│   └── requirements.txt
├── data/
│   ├── Dockerfile
│   ├── scripts/                 # pipeline_config, ingest*, preprocess*, validate, …
│   ├── raw/                     # gitignored
│   └── processed/               # gitignored
├── benchmark/                   # sentence_quality, sign_speak (see docs)
├── infrastructure/              # Terraform + k8s/
├── docs/                        # guides (index: docs/README.md)
├── .github/workflows/ci.yml
├── .github/scripts/merge_coverage_report.py
├── Dockerfile
├── docker-compose.yml
└── package.json                 # monorepo root metadata only
```

---

## Architecture

Typical **runtime** path (local demo or deployed API):

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Mobile (Expo)                                                           │
│  • Screens: app/index.tsx, camera.tsx, history.tsx                       │
│  • API client: services/api.ts → EXPO_PUBLIC_API_URL or app.json extra   │
│  • History: AsyncStorage on device only                                  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  HTTPS, multipart (mp4/mov)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FastAPI (backend/app/main.py)                                           │
│  • GET  /health — liveness                                               │
│  • GET  /ready — readiness + model_loaded                                │
│  • POST /api/v1/predict — field `file` → PredictionResponse              │
│  • POST /api/v1/predict/sentence — repeated `files` (order, ≤12)         │
└───────────────┬────────────────────────────────────────┬─────────────────┘
                │                                        │
                │  per clip                              │  multi-clip only
                ▼                                        ▼
┌─────────────────────────────────┐    ┌──────────────────────────────────────┐
│  preprocessing.py               │    │  beam_search.py + gloss_lm.py        │
│  video bytes → (1,3,64,224,224) │    │  GlossBeamLM from data/gloss_lm.json │
│  [-1,1] normalization           │    │  beam_size, lm_weight (query params) │
└───────────────┬─────────────────┘    └──────────────────┬───────────────────┘
                │                                         │
                └────────────────┬────────────────────────┘
                                 ▼
                ┌─────────────────────────────┐
                │  model_service.py           │
                │  ml.i3d_msft.pytorch_i3d    │
                │  Inception I3D, 856 glosses │
                │  predict / predict_batch    │
                └──────────────┬──────────────┘
                               │
                               │  optional: GLOSS_ENGLISH_MODE
                               ▼
                ┌────────────────────────────┐
                │  gloss_to_english.py       │
                │  (+ t5 / Bedrock modules)  │
                └────────────────────────────┘

  Model weights: local model_cache/ or download from S3 (see app/config.py defaults)
  Label map: ml/i3d_label_map_mvp-sft-full-v1.json
```

**Same repository, separate workflows:** `data/scripts/` (ingest & preprocess), `ml/modal_train_i3d.py` (cloud training), `infrastructure/` (Terraform), `benchmark/` (offline evaluation helpers). They are not required to run the app against an existing API.

---

## Deployed model

| Item | Detail |
|------|--------|
| Architecture | Inception I3D — `ml/i3d_msft/pytorch_i3d.py` |
| Input tensor | `(1, 3, 64, 224, 224)`, RGB, normalized to **[-1, 1]** |
| Classes | **856** glosses — `ml/i3d_label_map_mvp-sft-full-v1.json` |
| Inference preprocessing | `backend/app/services/preprocessing.py` (must match training) |
| Default weights | S3 path from `Settings` / `.env` — downloaded on first start if missing |

---

## Training datasets

These corpora feed the **data pipeline** (`data/scripts/`) and **I3D training** in `ml/`. The **deployed classifier** uses a **fixed 856-gloss** label map (see [Deployed model](#deployed-model)); training may merge or filter classes from the sources below.

| Dataset | Role here | What it is | Approx. scale |
|---------|-----------|------------|---------------|
| **ASL Citizen** | **Primary** — main supervised signal and signer-aware splits | Crowdsourced **isolated-sign** RGB videos (dictionary-style clips) from many contributors; varied backgrounds and capture conditions; intended for isolated sign recognition and retrieval research | ~**2.7k** glosses · ~**83k** videos · **52** signers |
| **WLASL** | **Supplementary** training | **Word-level** isolated ASL benchmark: short clips per English gloss/lemma, mix of studio and in-the-wild footage; widely used for word-level SLR baselines | ~**2k** glosses · ~**21k** videos · **100+** signers |
| **MS-ASL** | **Supplementary** training | Microsoft **large-vocabulary** isolated-sign dataset in **unconstrained** real-world settings (RGB only); emphasizes scale and signer-independent test conditions | ~**1k** gloss classes · ~**25k** videos · **200+** signers |

Counts are **order-of-magnitude** from the respective papers/projects; see each dataset’s documentation for exact numbers, splits, and download rules.

### References (data sources)

Eye Hear U builds on publicly released corpora. Cite the original publications (and respect each dataset’s license and terms) if you use this codebase in research or redistribute derived data.

| Dataset | Reference | Links |
|---------|-----------|--------|
| **ASL Citizen** | Desai, A., *et al.* “ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition.” *NeurIPS 2023* Datasets and Benchmarks Track. | [Paper (arXiv:2304.05934)](https://arxiv.org/abs/2304.05934) · [Project](https://www.microsoft.com/en-us/research/project/asl-citizen/) |
| **WLASL** | Li, D., Rodriguez, C., Yu, X., & Li, H. “Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison.” *WACV*, 2020. | [Paper (arXiv:1910.11006)](https://arxiv.org/abs/1910.11006) · [Project](https://dxli94.github.io/WLASL/) |
| **MS-ASL** | Vaezi Joze, H., & Koller, O. “MS-ASL: A Large-Scale Data Set and a Benchmark for Understanding American Sign Language.” *BMVC*, 2019. | [Paper (arXiv:1812.01053)](https://arxiv.org/abs/1812.01053) · [Microsoft Research](https://www.microsoft.com/en-us/research/project/ms-asl/) |

---

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Index of all guides |
| [User guide](docs/USER_GUIDE.md) | How to use the mobile app |
| [Developer guide](docs/DEVELOPER_GUIDE.md) | Day-to-day development, URLs, code map |
| [ASL translation pipeline](docs/ASL_TRANSLATION_PIPELINE.md) | Single vs multi-clip, beam, LM, English modes |
| [Testing](docs/TESTING.md) | pytest/Jest, coverage, CI behavior |
| [Production](docs/PRODUCTION.md) | AWS, containers, security checklist |
| [Preprocessing](docs/PREPROCESSING.md) | I3D input pipeline rationale |
| [Evaluation](docs/EVALUATION.md) | Metrics and evaluation workflows |
| [Benchmarking](docs/BENCHMARKING.md) | Reproducing benchmark numbers |
| [I3D training (S3)](docs/I3D_TRAINING_S3_REPRODUCTION.md) | Splits, S3, training reproduction |
| [Modal / AWS SFT migration](docs/MODAL_AWS_SFT_MIGRATION.md) | Account migration, Modal, warm-start |

---

## Development workflows

Optional paths after [Getting started](#getting-started): **automated tests** (parity with CI), **dataset ingest**, **I3D training on Modal**, **cloud / container deployment**, and **offline evaluation**. Install dependencies per component as in Getting started before running commands below.

### Tests

CI enforces **100% coverage** on scoped packages; run the same suites locally. Details, configs, and artifact layout: [Testing](docs/TESTING.md).

```bash
cd backend && export PYTHONPATH=.. && pytest tests/ -v --cov=app --cov-fail-under=100
cd ml && python3 -m pytest tests/ -v --cov=i3d_msft --cov=modal_train_i3d --cov-config=.coveragerc --cov-fail-under=100
cd mobile && npx jest --coverage --ci
```

### Data pipeline

Scripts in **`data/scripts/`** ingest **ASL Citizen**, **WLASL**, and **MS-ASL**, preprocess clips, validate, and plan I3D splits. Outputs live under **`data/raw/`** and **`data/processed/`** (gitignored). Overview of paths: [Repository structure](#repository-structure); day-to-day notes: [Developer guide](docs/DEVELOPER_GUIDE.md).

### ML training (I3D on Modal)

GPU training is driven by **`ml/modal_train_i3d.py`** (Modal). For S3 split plans, bucket layout, and reproduction: [I3D training (S3)](docs/I3D_TRAINING_S3_REPRODUCTION.md). For AWS migration, warm-start checkpoints, and Modal setup: [Modal / AWS SFT migration](docs/MODAL_AWS_SFT_MIGRATION.md).

```bash
modal run ml/modal_train_i3d.py --help
```

### Infrastructure and containers

| Goal | Command |
|------|---------|
| **AWS (Terraform)** | `cd infrastructure && terraform init && terraform apply -var-file=environments/dev.tfvars` |
| **Kubernetes** | `kubectl apply -k infrastructure/k8s/` |
| **Docker Compose** (API image) | `docker compose up --build` (repo root) |

Checklists, TLS, secrets, and ops notes: [Production](docs/PRODUCTION.md).

### Benchmarks and evaluation

Classifier benchmarks and metrics: [Benchmarking](docs/BENCHMARKING.md), [Evaluation](docs/EVALUATION.md). Why inference preprocessing must match training: [Preprocessing](docs/PREPROCESSING.md).

**All documentation:** [docs/README.md](docs/README.md).

---

## Continuous integration (CI)

On each push to **`main`** or pull request, GitHub Actions runs **backend**, **ML**, and **mobile** tests with **100% coverage thresholds** on the scoped packages. A follow-up job posts a **PR comment** on pull requests and may open a **separate PR** to refresh the table below after pushes to **`main`** when protected branches disallow direct bot pushes. See [Testing](docs/TESTING.md).

<!-- COVERAGE_TABLE_START -->
![Backend coverage](https://img.shields.io/badge/coverage%3A%20Backend-100%25-brightgreen) ![ML coverage](https://img.shields.io/badge/coverage%3A%20ML-100%25-brightgreen) ![Mobile coverage](https://img.shields.io/badge/coverage%3A%20Mobile-100%25-brightgreen)

| Component | Lines | Branches |
|-----------|-------|----------|
| Backend | **100%** | **100%** |
| ML | **100%** | **100%** |
| Mobile | **100%** | **100%** |

<sub>Last CI update: (overwritten on each push to `main`)</sub>
<!-- COVERAGE_TABLE_END -->

**Do not edit the table or badges between the HTML comments by hand** — automation replaces that block. You may edit the surrounding section text.

---

## Disclaimer

Use responsibly. Output quality depends on lighting, framing, and model limits; this software is **not** a substitute for a qualified human interpreter where one is required.
