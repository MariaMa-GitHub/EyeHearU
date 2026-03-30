## Eye Hear U

[![CI](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml)

**ASL on camera → text and speech on iPhone** — one sign at a time, or a short phrase from several clips in order.

**Eye Hear U** is an **iOS** app (Expo) with a **backend API**. You record **American Sign Language**; the app shows a **written result** and can **read it aloud**. It is built as an **assistive aid**—useful for practice or quick help—not a replacement for a **certified interpreter**.

Each clip is scored by a **video model** (trained on many signing examples) against a fixed **vocabulary of glosses**—short, English-like labels for signs. **Single-sign** mode focuses on the **best-matching label** (and a few alternates). **Multi-sign** mode also chooses a **sensible sequence** across clips before turning labels into the line you see. By default that line is a **joined, lightly cleaned phrase**; the API can be configured so a **separate language model** (running **locally** or via **AWS**) **rephrases** the main multi-sign result into **smoother, more conversational English** when you want that trade-off. Quality still depends on video, lighting, and the model—see **[ASL translation pipeline](docs/ASL_TRANSLATION_PIPELINE.md)** for behavior, limits, and `.env` options such as **`GLOSS_ENGLISH_MODE`**.

Infrastructure is provisioned on **AWS** (S3, ECS, etc.) via **Terraform**.

---

<details open>
<summary><strong>CI test coverage</strong> (auto-updated on <code>main</code>)</summary>

<!-- COVERAGE_TABLE_START -->
![Backend coverage](https://img.shields.io/badge/coverage%3A%20Backend-100%25-brightgreen) ![ML coverage](https://img.shields.io/badge/coverage%3A%20ML-100%25-brightgreen) ![Mobile coverage](https://img.shields.io/badge/coverage%3A%20Mobile-100%25-brightgreen)

| Component | Lines | Branches |
|-----------|-------|----------|
| Backend | **100%** | **100%** |
| ML | **100%** | **100%** |
| Mobile | **100%** | **100%** |

<sub>Last CI update: (overwritten on each push to <code>main</code>)</sub>
<!-- COVERAGE_TABLE_END -->

</details>

---

## Table of contents

| Area | Links |
|:-----|:------|
| **Docs** | [Documentation](#documentation) |
| **System** | [Architecture overview](#architecture-overview) → [Model](#model) · [Datasets](#datasets) · [Project structure](#project-structure) |
| **Runbook** | [How to run this repository](#how-to-run-this-repository) — [Prerequisites](#prerequisites) · [Minimal path (inference + app)](#minimal-path-inference--app) |
| **Components** | [Quick start](#quick-start) — [Backend](#backend-inference-api) · [Mobile app](#mobile-app) · [Data pipeline](#data-pipeline) · [I3D training (Modal GPU)](#i3d-training-modal-gpu) · [Infrastructure (Terraform)](#infrastructure-terraform) · [Kubernetes deployment](#kubernetes-deployment-alternative-to-ecs) · [Docker](#docker) |
| **Quality** | [Testing](#testing) |

---

## Documentation

| Audience                                       | Document                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| End users                                      | [User guide](docs/USER_GUIDE.md)                                       |
| Developers                                     | [Developer guide](docs/DEVELOPER_GUIDE.md)                             |
| Single vs multi-clip pipeline & accuracy scope | [ASL translation pipeline](docs/ASL_TRANSLATION_PIPELINE.md)           |
| Testing & coverage                             | [Testing](docs/TESTING.md)                                             |
| Production deployment                          | [Production](docs/PRODUCTION.md)                                       |
| Inference preprocessing                        | [Preprocessing (I3D)](docs/PREPROCESSING.md)                           |
| Evaluation metrics guide                       | [Evaluation](docs/EVALUATION.md)                                       |
| Benchmarking & evaluation                      | [Benchmarking](docs/BENCHMARKING.md)                                   |
| I3D training (S3 reproduction)                 | [I3D training — S3 reproduction](docs/I3D_TRAINING_S3_REPRODUCTION.md) |
| Modal / AWS / SFT migration                    | [Modal & AWS SFT migration](docs/MODAL_AWS_SFT_MIGRATION.md)           |

**Coverage scope:** CI enforces **100%** lines and branches on backend `app/` and ML (`i3d_msft` + `modal_train_i3d`), and **100%** statements, branches, lines, and functions on mobile `app/` + `services/`. The **CI test coverage** panel is refreshed on **`main`** by `github-actions[bot]` (`[skip ci]`). See [Testing](docs/TESTING.md).

---

## Architecture overview

```
┌──────────────────────┐        ┌──────────────────────────────────────┐
│   Mobile App         │  HTTP  │   Inference API                      │
│   (React Native /    │───────▶│   (FastAPI on ECS Fargate behind ALB)│
│    Expo)             │        │                                      │
│                      │        │   POST /api/v1/predict               │
│  - Camera + library  │        │   → one clip → top gloss + top_k     │
│  - Single / multi-   │◀───────│                                      │
│    sign modes        │  JSON  │   POST /api/v1/predict/sentence      │
│  - TTS, local history│        │   → N clips → beam + gloss LM →     │
│  - SignASL previews  │        │      best_glosses + english line*    │
│                      │        │   Loads I3D + gloss_lm.json @ startup│
└──────────────────────┘        └─────────────────┬────────────────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │   ML Model          │
                                       │   (PyTorch)         │
                                       │                     │
                                       │   Inception I3D     │
                                       │   (spatiotemporal   │
                                       │    3D CNN)          │
                                       │        ↓            │
                                       │   Classification    │
                                       │   (856 glosses)     │
                                       └─────────────────────┘

  ┌─────────────────────┐   ┌────────────────────┐   ┌──────────────────┐
  │   Amazon S3         │   │   Firebase         │   │   CloudWatch     │
  │   (Data Lake)       │   │   (Firestore)      │   │   (Logs/Alerts)  │
  │                     │   │                    │   │                  │
  │   raw/ → processed/ │   │   - Translation    │   │   - Pipeline     │
  │   → models/         │   │     history        │   │     metrics      │
  │                     │   │   - User feedback  │   │   - API latency  │
  └─────────────────────┘   └────────────────────┘   └──────────────────┘
```

\* **`english`** on `/predict/sentence`: formatted gloss line under **`GLOSS_ENGLISH_MODE=rule`**, or an optional **FLAN-T5** / **Bedrock** rewrite when configured (see [ASL translation pipeline](docs/ASL_TRANSLATION_PIPELINE.md)).

### Model

The deployed model is **Microsoft's Inception I3D** (spatiotemporal 3D CNN), fine-tuned on 856 ASL gloss classes from the ASL Citizen dataset. Key specifications:

| Property      | Value                                                                                |
| ------------- | ------------------------------------------------------------------------------------ |
| Architecture  | Inception I3D (`ml/i3d_msft/pytorch_i3d.py`)                                         |
| Input         | `(1, 3, 64, 224, 224)` — 64 RGB frames at 224x224                                    |
| Normalization | `[-1, 1]` pixel range                                                                |
| Output        | 856-class logits, temporally max-pooled                                              |
| Label map     | `ml/i3d_label_map_mvp-sft-full-v1.json`                                              |
| Weights       | S3: `s3://eye-hear-u-public-data-ca1/models/i3d/...` (auto-downloaded by backend)    |
| Preprocessing | Short-side-256 resize, center-crop 224x224 (`backend/app/services/preprocessing.py`) |

## Datasets

| Dataset         | Role                         | Size                                   |
| --------------- | ---------------------------- | -------------------------------------- |
| **ASL Citizen** | Primary (train / val / test) | 2,731 glosses, ~83K videos, 52 signers |
| **WLASL**       | Supplementary training        | 2,000 glosses, ~21K videos             |
| **MS-ASL**      | Supplementary training       | 1,000 glosses, ~25K videos             |

## Project structure

```
.
├── backend/                  # FastAPI inference API
│   ├── app/
│   │   ├── main.py           # App entrypoint, lifespan loads I3D + gloss LM
│   │   ├── config.py         # S3 bucket, model path, label map, gloss_lm_path
│   │   ├── routers/          # health.py, predict.py (/predict, /predict/sentence)
│   │   ├── schemas/          # Pydantic models
│   │   └── services/         # model_service, preprocessing, gloss_lm, beam_search,
│   │                         # gloss_to_english (+ optional T5 / Bedrock), lm_builder, firebase
│   ├── data/
│   │   └── gloss_lm.json     # Gloss n-gram stats (rebuild: scripts/build_gloss_lm.py)
│   ├── scripts/
│   │   └── build_gloss_lm.py # Offline LM builder from label map + optional gloss lines
│   ├── tests/                # ~156 tests, 100% line+branch on app/
│   └── requirements.txt
│
├── mobile/                   # React Native (Expo) mobile app
│   ├── app/                  # home, camera (single + multi-sign, TTS, SignASL previews), history
│   ├── __tests__/            # Jest — 100% coverage on collected paths (package.json)
│   ├── services/api.ts     # predictSign (/predict), predictSentence (/predict/sentence)
│   └── package.json
│
├── ml/                       # Machine learning code
│   ├── i3d_msft/             # Inception I3D — the deployed model
│   │   ├── pytorch_i3d.py    # InceptionI3d architecture (from Microsoft)
│   │   ├── videotransforms.py
│   │   ├── export_label_map.py
│   │   ├── train.py          # I3D training with S3 data + Modal GPU
│   │   ├── evaluate.py       # I3D evaluation (top-k, MRR, DCG, confusion)
│   │   ├── dataset.py        # ASLCitizenI3DDataset (64-frame, [-1,1] norm)
│   │   ├── s3_data.py        # S3 sync helpers (splits, clips)
│   │   └── build_label_map_artifacts.py  # Rebuild label map from training
│   ├── i3d_label_map_mvp-sft-full-v1.json  # 856-class label map (v4)
│   ├── modal_train_i3d.py    # Modal GPU wrapper for cloud training
│   ├── tests/                # ~194 tests; 100% line+branch on i3d_msft + modal_train_i3d (see TESTING.md)
│   └── requirements.txt
│
├── data/                     # Data pipeline
│   ├── Dockerfile            # Pipeline container image
│   ├── scripts/
│   │   ├── pipeline_config.py          # Shared config (local + S3)
│   │   ├── ingest_asl_citizen.py
│   │   ├── ingest_wlasl.py
│   │   ├── ingest_msasl.py
│   │   ├── preprocess_clips.py
│   │   ├── build_unified_dataset.py
│   │   ├── validate.py
│   │   ├── plan_i3d_splits.py          # Versioned S3 split plans
│   │   ├── prepare_i3d_from_s3.py      # Download & prepare I3D data
│   │   └── requirements.txt
│   ├── raw/                  # Raw video files (gitignored)
│   └── processed/            # Processed clips + metadata (gitignored)
│
├── benchmark/                # Offline benchmarks (sentence quality, sign_speak; see BENCHMARKING.md)
│   ├── sentence_quality/
│   └── sign_speak/
│
├── infrastructure/           # Infrastructure as Code
│   ├── main.tf               # Root module, provider config
│   ├── variables.tf
│   ├── outputs.tf
│   ├── environments/
│   │   ├── dev.tfvars
│   │   ├── staging.tfvars
│   │   └── prod.tfvars
│   ├── modules/              # Terraform modules
│   │   ├── s3/               # Data lake bucket
│   │   ├── ecr/              # Container registries
│   │   ├── batch/            # Pipeline job compute
│   │   ├── ecs/              # API cluster + ALB
│   │   ├── iam/              # Roles and policies
│   │   ├── networking/       # VPC, subnets, NAT
│   │   └── monitoring/       # CloudWatch, SNS alerts
│   └── k8s/                  # Kubernetes manifests
│       ├── namespace.yaml
│       ├── deployment.yaml   # API Deployment (2 replicas)
│       ├── service.yaml      # ClusterIP Service
│       ├── ingress.yaml      # ALB Ingress
│       ├── configmap.yaml    # Environment configuration
│       └── hpa.yaml          # Horizontal Pod Autoscaler
│
├── docs/
│   ├── USER_GUIDE.md         # End-user guide
│   ├── DEVELOPER_GUIDE.md    # Setup and day-to-day development
│   ├── ASL_TRANSLATION_PIPELINE.md  # Single vs multi-clip, beam+LM, output semantics
│   ├── TESTING.md            # Tests, coverage, CI
│   ├── PRODUCTION.md         # Production deployment
│   ├── PREPROCESSING.md      # I3D inference preprocessing
│   ├── EVALUATION.md         # How to generate evaluation metrics
│   ├── BENCHMARKING.md       # Evaluation metrics and reproduction
│   ├── I3D_TRAINING_S3_REPRODUCTION.md  # I3D training reproduction with S3
│   └── MODAL_AWS_SFT_MIGRATION.md       # AWS / Modal / SFT migration playbook
│
├── .github/workflows/ci.yml  # GitHub Actions CI (backend, ML, mobile)
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## How to run this repository

Use this order for a full local loop: **backend API** (inference) → **mobile app** (Expo). Training, data ingest, Terraform, and benchmarks are optional paths below.

### Prerequisites

| Requirement      | Notes                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| **Python 3.11+** | CI uses 3.11. On macOS, use `python3` if `python` is not on your `PATH`.                               |
| **Node.js 20+**  | Matches CI; use `npm ci --legacy-peer-deps` in `mobile/` for a clean install.                          |
| **pip / venv**   | Recommended per component (`backend/`, `ml/`, `data/scripts/` each have their own `requirements.txt`). |

### Minimal path (inference + app)

1. **Start the API** (from `backend/`, with repo root on `PYTHONPATH` so `ml` resolves):

   ```bash
   cd backend
   python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
   pip install -r requirements.txt
   cp .env.example .env
   export PYTHONPATH=..    # parent directory = repo root
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **In another terminal, start the mobile client:**

   ```bash
   cd mobile
   npm install --legacy-peer-deps
   cp .env.example .env   # if present; set EXPO_PUBLIC_API_URL to your machine’s LAN IP :8000
   npx expo start
   ```

3. **Sanity-check the API** (optional):

   ```bash
   curl http://localhost:8000/health
   ```

4. **Run automated tests** (optional, same gates as CI):

   ```bash
   # Backend (repo root on PYTHONPATH)
   cd backend && export PYTHONPATH=.. && pytest tests/ -v --cov=app --cov-fail-under=100

   # ML
   cd ml && python3 -m pytest tests/ -v --cov=i3d_msft --cov=modal_train_i3d --cov-config=.coveragerc --cov-fail-under=100

   # Mobile
   cd mobile && npx jest --coverage --ci
   ```

See [Developer guide](docs/DEVELOPER_GUIDE.md) for LAN vs tunnel, simulators, and `.env` details.

---

## Quick start

### Backend (Inference API)

The backend serves the I3D model. On first startup it downloads the checkpoint from S3 automatically.

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env if needed (MODEL_DEVICE, LABEL_MAP_PATH, GLOSS_ENGLISH_MODE, optional BEDROCK_*)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test the API:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict -F "file=@sample.mp4"
# Multi-clip sentence (beam search + gloss LM): same field name `files` repeated
curl -X POST "http://localhost:8000/api/v1/predict/sentence?beam_size=8&lm_weight=1" \
  -F "files=@clip1.mp4" -F "files=@clip2.mp4"
```

### Mobile app

```bash
cd mobile
npm install --legacy-peer-deps
npx expo start
```

Set `EXPO_PUBLIC_API_URL` in `mobile/.env` to point to the backend (e.g., `http://192.168.x.x:8000`). See [Developer guide](docs/DEVELOPER_GUIDE.md) for LAN / tunnel setup.

### Data pipeline

```bash
cd data/scripts
pip install -r requirements.txt

export PIPELINE_ENV=local
python ingest_asl_citizen.py
python ingest_wlasl.py
python ingest_msasl.py
python preprocess_clips.py
python build_unified_dataset.py
python validate.py
```

### I3D training (Modal GPU)

```bash
pip install modal
modal setup  # one-time auth
# Smoke test (1 epoch, 200 clips)
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 1 --clip-limit 200
# Full training
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 20
```

See [I3D training — S3 reproduction](docs/I3D_TRAINING_S3_REPRODUCTION.md) and [Modal & AWS SFT migration](docs/MODAL_AWS_SFT_MIGRATION.md) for details.

### Infrastructure (Terraform)

```bash
cd infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars
```

### Kubernetes deployment (alternative to ECS)

```bash
kubectl apply -k infrastructure/k8s/
```

### Docker

```bash
docker compose up --build
```

## Testing

CI runs **three** test jobs in parallel, then a **`coverage-report`** job that posts a **sticky PR comment** (same-repo PRs) and refreshes the **README** coverage block on pushes to **`main`** / **`master`** (see [Testing](docs/TESTING.md)).

| Job     | Tests (approx.) | Coverage                                                      | Enforced                                          |
| ------- | --------------- | ------------------------------------------------------------- | ------------------------------------------------- |
| Backend | ~156 pytest     | 100% line + branch on `app/`                                  | `--cov-fail-under=100`                            |
| ML      | ~194 pytest     | 100% line + branch on `i3d_msft` + `modal_train_i3d`         | `--cov-fail-under=100`                            |
| Mobile  | ~104 Jest       | 100% statements, branches, lines, functions (collected paths) | Jest `coverageThreshold` in `mobile/package.json` |

Run locally:

```bash
# Backend (PYTHONPATH must include repo root)
cd backend && export PYTHONPATH=.. && pytest tests/ -v --cov=app --cov-fail-under=100

# ML (scoped coverage — see docs/TESTING.md)
cd ml && python3 -m pytest tests/ -v \
  --cov=i3d_msft --cov=modal_train_i3d \
  --cov-config=.coveragerc --cov-fail-under=100

# Mobile
cd mobile && npx jest --coverage --ci
```

See [Testing](docs/TESTING.md) for full details and [Evaluation](docs/EVALUATION.md) for generating metrics for reports.
