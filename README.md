## Eye Hear U

[![CI](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MariaMa-GitHub/EyeHearU/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MariaMa-GitHub/EyeHearU/branch/main/graph/badge.svg)](https://codecov.io/gh/MariaMa-GitHub/EyeHearU)

**Real-time ASL-to-English translation on iOS** — one sign at a time.

Eye Hear U translates isolated American Sign Language (ASL) signs into English text and speech using a mobile app, a backend inference API, and a video classifier trained on public ASL datasets. Infrastructure is provisioned on AWS via Terraform.

### Documentation

| Audience | Document |
|----------|----------|
| End users | [User guide](docs/USER_GUIDE.md) |
| Developers | [Developer guide](docs/DEVELOPER_GUIDE.md) |
| Testing & coverage | [Testing](docs/TESTING.md) |
| Production deployment | [Production](docs/PRODUCTION.md) |
| Inference preprocessing | [Preprocessing (I3D)](docs/PREPROCESSING.md) |
| Evaluation metrics guide | [Evaluation](docs/EVALUATION.md) |
| Benchmarking & evaluation | [Benchmarking](docs/BENCHMARKING.md) |
| I3D training (S3 repro) | [I3D S3 Repro Guide](docs/i3d_s3_repro_guide.md) |
| Modal / AWS migration | [Ops Migration Tutorial](docs/ops_migration_modal_sft_tutorial.md) |
| Profiling analysis | [Profiling](docs/PROFILING.md) |

**Codecov:** Register the repository at [codecov.io](https://about.codecov.io/) and add the `CODECOV_TOKEN` secret under GitHub → Settings → Secrets → Actions so PR comments and the badge update automatically. CI enforces **100%** backend and ML line/branch coverage independently of Codecov.

---

## Architecture Overview

```
┌──────────────────────┐        ┌──────────────────────────────────────┐
│   Mobile App         │  HTTP  │   Inference API                      │
│   (React Native /    │───────▶│   (FastAPI on ECS Fargate behind ALB)│
│    Expo)             │        │                                      │
│                      │        │   POST /api/v1/predict               │
│  - Camera capture    │◀───────│   → sign label + confidence          │
│  - Display results   │  JSON  │                                      │
│  - Text-to-speech    │        │   Loads I3D model from S3 at startup │
│  - Translation       │        └─────────────────┬────────────────────┘
│    history           │                          │
└──────────────────────┘                          │
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

### Model

The deployed model is **Microsoft's Inception I3D** (spatiotemporal 3D CNN), fine-tuned on 856 ASL gloss classes from the ASL Citizen dataset. Key specifications:

| Property | Value |
|----------|-------|
| Architecture | Inception I3D (`ml/i3d_msft/pytorch_i3d.py`) |
| Input | `(1, 3, 64, 224, 224)` — 64 RGB frames at 224x224 |
| Normalization | `[-1, 1]` pixel range |
| Output | 856-class logits, temporally max-pooled |
| Label map | `ml/i3d_label_map_mvp-sft-full-v1.json` |
| Weights | S3: `s3://eye-hear-u-public-data-ca1/models/i3d/...` (auto-downloaded by backend) |
| Preprocessing | Short-side-256 resize, center-crop 224x224 (`backend/app/services/preprocessing.py`) |

## Datasets

| Dataset | Role | Size |
|---|---|---|
| **ASL Citizen** | Primary (train / val / test) | 2,731 glosses, ~83K videos, 52 signers |
| **WLASL** | Supplementary training | 2,000 glosses, ~21K videos |
| **MS-ASL** | Supplementary training | 1,000 glosses, ~25K videos |

## Project Structure

```
.
├── backend/                  # FastAPI inference API
│   ├── app/
│   │   ├── main.py           # App entrypoint, lifespan loads I3D model
│   │   ├── config.py         # S3 bucket, model path, label map path
│   │   ├── routers/          # health.py, predict.py
│   │   ├── schemas/          # Pydantic models
│   │   └── services/         # model_service (I3D), preprocessing, firebase
│   ├── tests/                # 82 tests, 100% coverage
│   └── requirements.txt
│
├── mobile/                   # React Native (Expo) mobile app
│   ├── app/                  # _layout, index, camera, history
│   ├── __tests__/            # 59 tests, 100% line coverage
│   ├── services/api.ts       # API client for /predict endpoint
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
│   ├── profiling/            # cProfile analysis of 5 key functions
│   ├── tests/                # 190+ unit tests, 100% coverage
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
│   ├── TESTING.md            # Tests, coverage, CI
│   ├── PRODUCTION.md         # Production deployment
│   ├── PREPROCESSING.md      # I3D inference preprocessing
│   ├── EVALUATION.md         # How to generate evaluation metrics
│   ├── BENCHMARKING.md       # Evaluation metrics and reproduction
│   ├── PROFILING.md          # cProfile analysis of 5 key functions
│   ├── i3d_s3_repro_guide.md # I3D training reproduction with S3
│   └── ops_migration_modal_sft_tutorial.md  # AWS/Modal migration
│
├── .github/workflows/ci.yml  # GitHub Actions CI (backend, ML, mobile)
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## Quick Start

### Backend (Inference API)

The backend serves the I3D model. On first startup it downloads the checkpoint from S3 automatically.

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env if needed (MODEL_DEVICE, LABEL_MAP_PATH, etc.)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Test the API:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict -F "file=@sample.mp4"
```

### Mobile App

```bash
cd mobile
npm install --legacy-peer-deps
npx expo start
```

Set `EXPO_PUBLIC_API_URL` in `mobile/.env` to point to the backend (e.g., `http://192.168.x.x:8000`). See [Developer guide](docs/DEVELOPER_GUIDE.md) for LAN / tunnel setup.

### Data Pipeline

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

### I3D Training (Modal GPU)

```bash
pip install modal
modal setup  # one-time auth
# Smoke test (1 epoch, 200 clips)
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 1 --clip-limit 200
# Full training
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 20
```

See [I3D S3 Repro Guide](docs/i3d_s3_repro_guide.md) and [Ops Migration Tutorial](docs/ops_migration_modal_sft_tutorial.md) for details.

### Infrastructure (Terraform)

```bash
cd infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars
```

### Kubernetes Deployment (alternative to ECS)

```bash
kubectl apply -k infrastructure/k8s/
```

### Docker

```bash
docker compose up --build
```

## Testing

CI runs three parallel jobs on every push/PR to `main`:

| Job | Tests | Coverage | Enforced |
|-----|-------|----------|----------|
| Backend | 82 pytest | 100% line + branch | `--cov-fail-under=100` |
| ML | 190+ pytest | 100% line | `--cov-fail-under=100` |
| Mobile | 59 Jest | 100% function | Jest coverage thresholds |

Run locally:

```bash
# Backend
cd backend && pytest tests/ -v --cov=app --cov-fail-under=100

# ML
cd ml && python -m pytest tests/ -v --cov --cov-fail-under=100

# Mobile
cd mobile && npx jest --coverage
```

See [Testing](docs/TESTING.md) for full details and [Evaluation](docs/EVALUATION.md) for generating metrics for reports.
