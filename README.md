## Eye Hear U

**Real-time ASL-to-English translation on iOS** — one sign at a time.

Eye Hear U translates isolated American Sign Language (ASL) signs into English text and speech using a mobile app, a backend inference API, and a video classifier trained on public ASL datasets. Infrastructure is provisioned on AWS via Terraform.

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
│  - Text-to-speech    │        │   Loads model from S3 at startup     │
│  - Translation       │        └─────────────────┬────────────────────┘
│    history           │                          │
└──────────────────────┘                          │
                                       ┌──────────▼──────────┐
                                       │   ML Model          │
                                       │   (PyTorch)         │
                                       │                     │
                                       │   3D CNN backbone   │
                                       │   (R3D-18 /         │
                                       │    Kinetics-400)    │
                                       │        ↓            │
                                       │   Classification    │
                                       │   → ~2000+ classes  │
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

## Datasets

| Dataset | Role | Size |
|---|---|---|
| **ASL Citizen** | Primary (train / val / test) | 2,731 glosses, ~83K videos, 52 signers |
| **WLASL** | Supplementary training | 2,000 glosses, ~21K videos |
| **MS-ASL** | Supplementary training | 1,000 glosses, ~25K videos |

## Project Structure

```
.
├── backend/                  # FastAPI backend server
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/          # health.py, predict.py
│   │   ├── schemas/          # Pydantic models
│   │   └── services/         # model_service, preprocessing, firebase
│   └── requirements.txt
│
├── mobile/                   # React Native (Expo) mobile app
│   ├── app/                  # _layout, index, camera, history
│   ├── services/api.ts
│   └── package.json
│
├── ml/                       # Machine learning pipeline
│   ├── config.py             # Video classifier config
│   ├── models/classifier.py  # ASLVideoClassifier (3D CNN)
│   ├── training/
│   │   ├── train.py
│   │   └── dataset.py        # ASLVideoDataset (PyTorch)
│   ├── evaluation/
│   │   └── evaluate.py
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
│   │   └── requirements.txt
│   ├── raw/                  # Raw video files (gitignored)
│   └── processed/            # Processed clips + metadata (gitignored)
│
├── infrastructure/           # Terraform IaC
│   ├── main.tf               # Root module, provider config
│   ├── variables.tf
│   ├── outputs.tf
│   ├── environments/
│   │   ├── dev.tfvars
│   │   ├── staging.tfvars
│   │   └── prod.tfvars
│   └── modules/
│       ├── s3/               # Data lake bucket
│       ├── ecr/              # Container registries
│       ├── batch/            # Pipeline job compute
│       ├── ecs/              # API cluster + ALB
│       ├── iam/              # Roles and policies
│       ├── networking/       # VPC, subnets, NAT
│       └── monitoring/       # CloudWatch, SNS alerts
│
├── docs/
│   ├── architecture.md
│   ├── data_schema.md
│   ├── a2_writeup.md         # A2 Parts 1 & 2
│   └── data_pipeline.md      # A2 Part 3
│
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## Quick Start

### Infrastructure (Terraform)

```bash
cd infrastructure
terraform init
terraform apply -var-file=environments/dev.tfvars
```

### Data Pipeline

```bash
cd data/scripts
pip install -r requirements.txt

# Local run
export PIPELINE_ENV=local
python ingest_asl_citizen.py
python ingest_wlasl.py
python ingest_msasl.py
python preprocess_clips.py
python build_unified_dataset.py
python validate.py
```

### ML Training

```bash
cd ml
pip install -r requirements.txt
python -m training.train
python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

### Mobile App

```bash
cd mobile
npm install
npx expo start
```

---

For detailed documentation see `docs/architecture.md`, `docs/data_schema.md`, and `docs/data_pipeline.md`.
