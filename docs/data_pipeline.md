# Part Three: Data-Processing Pipeline

## 1. Pipeline Overview

Our data pipeline transforms three public ASL video datasets (ASL Citizen, WLASL, MS-ASL) into a single unified, preprocessed dataset that our video classifier (3D CNN pretrained on Kinetics-400) consumes during training and evaluation. The pipeline runs locally during development and on **AWS** in staging/production, orchestrated by **Apache Airflow** and deployed via **Terraform**.

The pipeline has four stages:

```
                          ┌─────────────────────────────┐
                          │     External Sources        │
                          │  ASL Citizen  WLASL  MS-ASL │
                          └──────────┬──────────────────┘
                                     │  download / manual upload
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  STAGE 1 — INGESTION                                                      │
 │  Containers: data-pipeline (Docker on AWS Batch / local)                  │
 │                                                                           │
 │  ingest_asl_citizen.py   ingest_wlasl.py   ingest_msasl.py                │
 │        │                       │                   │                      │
 │        ▼                       ▼                   ▼                      │
 │  ingested_asl_citizen.csv  ingested_wlasl.csv  ingested_msasl.csv         │
 │                                                                           │
 │  Reads from : S3  s3://{bucket}/raw/{source}/                             │
 │  Writes to  : S3  s3://{bucket}/processed/ingested_*.csv                  │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  STAGE 2 — PREPROCESSING                                                  │
 │  Container: data-pipeline (GPU-optional; CPU is sufficient)               │
 │  preprocess_clips.py                                                      │
 │                                                                           │
 │  For each ingested record:                                                │
 │    1. Read raw video from S3 (or local)                                   │
 │    2. Trim to sign boundaries (frame or time annotations)                 │
 │    3. Discard if < 8 frames or > 4 s                                      │
 │    4. Uniformly sample to 16 frames                                       │
 │    5. Resize every frame to 224 × 224                                     │
 │    6. Write normalised .mp4 → S3  s3://{bucket}/processed/clips/          │
 │                                                                           │
 │  Output: processed_clips.csv                                              │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  STAGE 3 — DATASET BUILDING                                               │
 │  build_unified_dataset.py                                                 │
 │                                                                           │
 │  1. Filter glosses with < 5 training clips                                │
 │  2. Build label_map.json  (gloss → integer, sorted)                       │
 │  3. Compute dataset_stats.json (per-split, per-class, per-source)         │
 │  4. Upload artifacts → S3  s3://{bucket}/processed/                       │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  STAGE 4 — VALIDATION                                                     │
 │  validate.py                                                              │
 │                                                                           │
 │  • Check all clip files exist in S3                                       │
 │  • Spot-check frame count & resolution                                    │
 │  • Verify signer-disjoint splits (ASL Citizen)                            │
 │  • Confirm label_map ↔ CSV consistency                                    │
 │  • Publish pass/fail metrics → Amazon CloudWatch                          │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  DATA LAKE  (Amazon S3)                                                   │
 │                                                                           │
 │  s3://eye-hear-u-{env}-data/                                              │
 │    raw/asl_citizen/videos/  ← immutable originals                         │
 │    raw/wlasl/videos/                                                      │
 │    raw/msasl/videos/                                                      │
 │    processed/clips/{split}/{gloss}/*.mp4                                  │
 │    processed/label_map.json                                               │
 │    processed/dataset_stats.json                                           │
 │    processed/processed_clips.csv                                          │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    │  reads at training time
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  MODEL TRAINING  (Amazon SageMaker / EC2 GPU)                             │
 │                                                                           │
 │  • Pulls processed clips from S3                                          │
 │  • Trains ASLVideoClassifier (PyTorch)                                    │
 │  • Writes checkpoints → S3  s3://{bucket}/models/                         │
 │  • Logs metrics → CloudWatch / MLflow                                     │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    │
                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  INFERENCE API  (Amazon ECS Fargate behind ALB)                           │
 │                                                                           │
 │  • FastAPI container (from Amazon ECR)                                    │
 │  • Loads best_model.pt + label_map.json from S3 at startup                │
 │  • POST /api/v1/predict → sign label + confidence                         │
 │  • Logs predictions → Amazon CloudWatch + Firebase Firestore              │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Schemas

Full schemas are documented in [`docs/data_schema.md`](./data_schema.md). A summary:

### 2.1 Raw Ingested Records (per source)

| Field        | Type   | Note                                             |
|--------------|--------|--------------------------------------------------|
| clip_id      | string | Unique per source                                |
| gloss        | string | ASL sign label, lower-cased                      |
| signer_id    | string | Anonymous signer identifier                      |
| split        | string | `train` / `val` / `test`                         |
| source       | string | `asl_citizen` / `wlasl` / `msasl`                |
| frame_start  | int    | Start frame boundary (WLASL only)                |
| frame_end    | int    | End frame boundary (WLASL only, -1 = end)        |
| start_time   | float  | Start timestamp in seconds (MS-ASL only)         |
| end_time     | float  | End timestamp in seconds (MS-ASL only)           |
| src_path     | string | Path or S3 URI to raw video                      |

### 2.2 Processed Clips Record

| Field      | Type   | Value                           |
|------------|--------|---------------------------------|
| clip_id    | string | Same as ingested                |
| gloss      | string | Lower-cased label               |
| signer_id  | string | Signer identifier               |
| split      | string | `train`/`val`/`test`            |
| source     | string | Origin dataset                  |
| num_frames | int    | 16                              |
| height     | int    | 224                             |
| width      | int    | 224                             |
| clip_path  | string | Local path or S3 URI to `.mp4`  |

### 2.3 Label Map

Flat JSON: `{ "gloss": int_index, … }` — up to ~2,731 classes (ASL Citizen vocabulary) after filtering.

---

## 3. Technologies Used

### 3.1 Pipeline & Data Processing

| Component             | Technology                        | Why                                                                     |
|-----------------------|-----------------------------------|-------------------------------------------------------------------------|
| Video I/O             | **OpenCV (cv2)**                  | Cross-platform, pip-installable, handles mp4 read/write                 |
| Array manipulation    | **NumPy**                         | Standard for frame arrays and uniform sampling                          |
| HTTP downloads        | **Requests**                      | Lightweight, for fetching metadata JSONs from GitHub                    |
| Data formats          | **CSV + JSON**                    | Human-readable, versionable, no database needed                         |
| Pipeline scripting    | **Python 3.11+**                  | Same language as the ML stack; shared with PyTorch                      |
| ML data loading       | **PyTorch Dataset/DataLoader**    | Direct integration with the training loop                               |
| Containerisation      | **Docker**                        | Reproducible environments; required for AWS Batch and ECS               |
| Pipeline orchestration| **Apache Airflow**                | DAG-based scheduling; task dependencies; retries; logs; open source     |
| Dataset versioning    | **DVC (Data Version Control)**    | Tracks large data files alongside Git; supports S3 remotes              |
| Experiment tracking   | **MLflow**                        | Logs training metrics, hyperparameters, and model artifacts             |

### 3.2 AWS Cloud Services

| AWS Service              | Role in Pipeline                                                                  |
|--------------------------|-----------------------------------------------------------------------------------|
| **Amazon S3**            | Data lake for raw videos, processed clips, metadata CSVs, label maps, model checkpoints. Two-layer structure: `raw/` (immutable) and `processed/` (derived). S3 versioning enabled for disaster recovery. |
| **Amazon ECR**           | Docker container registry. Stores the `data-pipeline` image (for ingestion/preprocessing) and the `backend-api` image (FastAPI inference server). |
| **AWS Batch**            | Runs pipeline stages (ingestion, preprocessing, building, validation) as containerised batch jobs. Jobs pull images from ECR and read/write to S3. Automatically provisions and scales compute. |
| **Amazon SageMaker**     | Model training on GPU instances (e.g., `ml.g4dn.xlarge`). Reads processed clips from S3, trains the PyTorch video classifier, writes checkpoints back to S3. |
| **Amazon ECS (Fargate)** | Hosts the FastAPI inference API as a serverless container behind an Application Load Balancer. Loads the trained model from S3 at startup. |
| **Amazon CloudWatch**    | Centralised logging for all pipeline jobs, training runs, and API requests. Custom metrics for pipeline pass/fail status, validation results, and inference latency. Alarms trigger SNS notifications on failure. |
| **Amazon SNS**           | Sends email/Slack notifications on pipeline completion, validation failures, or training completion. |
| **AWS IAM**              | Fine-grained access control. Separate roles for: pipeline jobs (S3 read/write), training jobs (S3 + SageMaker), inference API (S3 read-only + CloudWatch write), and CI/CD. |
| **Amazon DynamoDB**      | (Optional) Stores pipeline run metadata (run ID, status, timestamps, clip counts) for auditing and debugging. Lightweight alternative to a full relational DB. |

### 3.3 Infrastructure as Code

| Tool            | Role                                                                                 |
|-----------------|--------------------------------------------------------------------------------------|
| **Terraform**   | Provisions all AWS resources (S3 buckets, ECR repos, Batch compute environments, ECS clusters, IAM roles, CloudWatch log groups, SNS topics). Supports multiple environments (`dev`, `staging`, `prod`) via Terraform workspaces or variable files. |
| **Docker Compose** | Local development: runs the full pipeline + backend + Airflow on a single machine. |

---

## 4. When the Pipelines Run and for Which Use Cases

### 4.1 One-Time Full Run (Dataset Preparation)

Run once at the start of model development, after the raw video files have been uploaded to S3.

**Locally:**
```bash
cd data/scripts

python ingest_asl_citizen.py
python ingest_wlasl.py
python ingest_msasl.py
python preprocess_clips.py
python build_unified_dataset.py
python validate.py
```

**On AWS (via Airflow DAG or manual AWS Batch submission):**
```bash
# Submit the full pipeline DAG
airflow dags trigger eye_hear_u_full_pipeline
```

The Airflow DAG defines the dependency chain: `ingest → preprocess → build → validate`, with each stage running as an AWS Batch job.

**Use case:** Produces the full processed data lake (on S3 or local disk) that the video classifier trains on.

### 4.2 Per-Source Incremental Run

When new videos are added to one source (e.g., additional WLASL videos are uploaded to S3):

```bash
python ingest_wlasl.py
python preprocess_clips.py --source wlasl
python build_unified_dataset.py
python validate.py
```

### 4.3 Training Time

The PyTorch `ASLVideoDataset` reads directly from the processed data lake (local or synced from S3):

```python
from ml.training.dataset import ASLVideoDataset

train_ds = ASLVideoDataset("data/processed", split="train", augment=True)
val_ds   = ASLVideoDataset("data/processed", split="val")
```

On AWS, SageMaker training jobs mount the S3 data as a local filesystem via SageMaker's built-in S3 integration (`s3://eye-hear-u-prod-data/processed/` → `/opt/ml/input/data/`).

### 4.4 Model Deployment

After training, the best checkpoint and label map are uploaded to S3:

```
s3://eye-hear-u-{env}-data/models/best_model.pt
s3://eye-hear-u-{env}-data/models/label_map.json
```

The ECS Fargate inference API downloads these at container startup.

### 4.5 Disaster Recovery

Because all infrastructure is defined in Terraform and all data lives in versioned S3 buckets:

1. `terraform destroy` tears down all AWS resources.
2. `terraform apply` recreates them identically.
3. S3 bucket versioning allows restoring deleted objects.
4. The pipeline can be re-run from raw data to fully reconstruct the processed layer.

---

## 5. Pipeline Code

All source code lives in `data/scripts/`:

| File                        | Stage         | Description                                                      |
|-----------------------------|---------------|------------------------------------------------------------------|
| `pipeline_config.py`        | Shared config | Paths (local + S3), URLs, preprocessing constants, environment   |
| `ingest_asl_citizen.py`     | Ingestion     | Parse ASL Citizen metadata, validate, write normalised CSV       |
| `ingest_wlasl.py`           | Ingestion     | Download WLASL JSON, parse instances, validate, write CSV        |
| `ingest_msasl.py`           | Ingestion     | Download MS-ASL JSONs, parse splits, validate, write CSV         |
| `preprocess_clips.py`       | Preprocessing | Trim → sample 16 frames → resize 224×224 → write .mp4           |
| `build_unified_dataset.py`  | Building      | Filter rare glosses, create label_map.json, dataset_stats.json   |
| `validate.py`               | Validation    | File existence, frame/resolution checks, signer-leak detection   |
| `requirements.txt`          | Dependencies  | opencv-python, numpy, requests, boto3                            |

Supporting files:

| File                        | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| `data/Dockerfile`           | Container image for all pipeline stages (Python 3.11 + OpenCV + boto3) |
| `ml/training/dataset.py`    | `ASLVideoDataset` — reads processed clips, returns (C,T,H,W) tensors |
| `ml/config.py`              | Dataclass config for model, data, and training hyperparameters   |

### 5.1 Key Design Decisions

**Uniform frame sampling:** We sample exactly 16 frames at equally spaced intervals per clip. This guarantees a fixed tensor shape `(C, 16, 224, 224)` regardless of the original clip duration, which is required for batching in the DataLoader and for 3D CNN input.

**Storing processed clips as `.mp4`:** We write the preprocessed frames back to `.mp4` rather than `.npy` arrays. This reduces disk and S3 storage by roughly 50× (video compression vs. raw float arrays) at the cost of a small decoding overhead per batch. Given that our dataset could be ~100K+ clips, storage savings outweigh the decoding cost.

**Signer-disjoint splits for ASL Citizen:** ASL Citizen ships with train/val/test splits that are stratified by signer — no signer appears in more than one split. We preserve these splits exactly, so validation and test accuracy measure generalization to **unseen signers**. This is critical because our app will be used by people the model has never seen.

**WLASL and MS-ASL used for training only:** Since we cannot guarantee signer-disjoint splits across datasets, supplementary clips from WLASL and MS-ASL are added **only to the training split**. Validation and test sets remain purely ASL Citizen to keep evaluation clean.

**Minimum 5 videos per gloss:** Glosses with fewer than 5 training clips are dropped, as the model cannot learn a reliable representation from so few examples.

**S3 as single source of truth:** All raw and processed data resides in S3 with versioning enabled. Local copies are caches — the pipeline can always be re-run from S3-hosted raw data. This also enables disaster recovery: even if the processed layer is deleted, it can be fully reconstructed from `raw/` + pipeline scripts.

**Environment isolation via S3 bucket naming:** Each environment (`dev`, `staging`, `prod`) has its own S3 bucket (`eye-hear-u-dev-data`, `eye-hear-u-staging-data`, `eye-hear-u-prod-data`), preventing accidental cross-environment contamination.

### 5.2 Augmentation Strategy

Augmentations are applied at training time (not during preprocessing) so that each epoch sees slightly different variants:

| Augmentation               | Probability | Details                                        |
|----------------------------|-------------|------------------------------------------------|
| Random temporal shift      | 50%         | Roll frames by ±1 position                     |
| Brightness/contrast jitter | 50%         | Alpha ∈ [0.8, 1.2], beta ∈ [-20, +20]          |
| Random spatial crop        | 50%         | Crop to 90% then resize back to 224×224        |

**No horizontal flip** — flipping would change a right-handed sign into a left-handed one, altering its meaning.

---

## 6. Data Lake Design (Amazon S3)

We adopt a **two-layer data lake** on Amazon S3, with environment-specific buckets:

```
s3://eye-hear-u-{env}-data/               ← one bucket per environment
│
├── raw/                                   ← LAYER 1: immutable originals
│   ├── asl_citizen/
│   │   ├── videos/*.mp4                   ← uploaded once, never modified
│   │   └── metadata.json
│   ├── wlasl/
│   │   ├── videos/*.mp4
│   │   └── WLASL_v0.3.json
│   └── msasl/
│       ├── videos/*.mp4
│       ├── classes.json
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── processed/                             ← LAYER 2: derived, reproducible
│   ├── clips/
│   │   ├── train/{gloss}/*.mp4            ← preprocessed fixed-size clips
│   │   ├── val/{gloss}/*.mp4
│   │   └── test/{gloss}/*.mp4
│   ├── ingested_asl_citizen.csv
│   ├── ingested_wlasl.csv
│   ├── ingested_msasl.csv
│   ├── processed_clips.csv
│   ├── label_map.json
│   └── dataset_stats.json
│
└── models/                                ← LAYER 3: trained artifacts
    ├── best_model.pt
    ├── label_map.json                     ← copy for inference
    └── training_config.json               ← snapshot of hyperparameters
```

**Layer 1 (raw/)** is append-only. We never modify or delete raw videos. S3 versioning is enabled so even accidental deletions can be recovered (disaster recovery).

**Layer 2 (processed/)** is fully derived and reproducible from Layer 1 + the pipeline scripts. If this layer is deleted, running the pipeline from scratch restores it completely.

**Layer 3 (models/)** stores trained model checkpoints. The inference API reads `best_model.pt` and `label_map.json` from here at startup.

### 6.1 S3 Bucket Configuration (managed by Terraform)

| Setting                  | Value                                     | Reason                                |
|--------------------------|-------------------------------------------|---------------------------------------|
| Bucket naming            | `eye-hear-u-{env}-data`                   | Environment isolation                 |
| Versioning               | Enabled                                   | Disaster recovery                     |
| Server-side encryption   | AES-256 (SSE-S3)                          | Data at rest protection               |
| Lifecycle rules          | Move `raw/` to S3 Infrequent Access after 30 days | Cost optimization for rarely accessed raw videos |
| Access                   | Private; IAM policies only                | Security                              |
| CORS                     | Disabled                                  | No browser-direct access              |

### 6.2 Local Development Mirror

During local development, the pipeline reads/writes to the local filesystem under `data/raw/` and `data/processed/`. The `pipeline_config.py` detects the `PIPELINE_ENV` environment variable:

- `PIPELINE_ENV=local` (default): uses local paths
- `PIPELINE_ENV=dev|staging|prod`: uses S3 paths via `boto3`

---

## 7. Infrastructure Architecture (AWS + Terraform)

The full infrastructure is provisioned by Terraform and supports three environments:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AWS Account                                      │
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                        │
│  │  dev         │   │  staging    │   │  prod       │  ← Terraform          │
│  │  environment │   │  environment│   │  environment│    workspaces         │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                        │
│         │                 │                 │                               │
│         ▼                 ▼                 ▼                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Per-Environment Resources                                           │   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐    │   │
│  │  │ S3 Bucket      │  │ ECR Repository │  │ CloudWatch Log Group │    │   │
│  │  │ (data lake)    │  │ (Docker imgs)  │  │ (pipeline + API logs)│    │   │
│  │  └────────┬───────┘  └────────┬───────┘  └──────────────────────┘    │   │
│  │           │                   │                                      │   │
│  │  ┌────────▼───────┐  ┌───────▼────────┐                              │   │
│  │  │ AWS Batch      │  │ ECS Fargate    │                              │   │
│  │  │ Compute Env    │  │ Cluster        │                              │   │
│  │  │ (pipeline jobs)│  │ (API service)  │                              │   │
│  │  └────────────────┘  └───────┬────────┘                              │   │
│  │                              │                                       │   │
│  │                     ┌────────▼────────┐                              │   │
│  │                     │ ALB             │                              │   │
│  │                     │ (load balancer) │                              │   │
│  │                     └─────────────────┘                              │   │
│  │                                                                      │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐    │   │
│  │  │ IAM Roles      │  │ SNS Topic      │  │ DynamoDB Table       │    │   │
│  │  │ (per-service)  │  │ (alerts)       │  │ (pipeline run logs)  │    │   │
│  │  └────────────────┘  └────────────────┘  └──────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Shared Resources (across environments)                              │   │
│  │                                                                      │   │
│  │  ┌─────────────────────────┐  ┌──────────────────────────────────┐   │   │
│  │  │ VPC + Subnets           │  │ SageMaker Notebook / Training    │   │   │
│  │  │ (networking)            │  │ (model development)              │   │   │
│  │  └─────────────────────────┘  └──────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Terraform Module Structure

```
infrastructure/
├── main.tf                  # Provider config, backend (S3 state), locals
├── variables.tf             # Input variables (environment, region, etc.)
├── outputs.tf               # Exported resource IDs and URLs
├── terraform.tfvars         # Default variable values
│
├── environments/
│   ├── dev.tfvars           # Dev overrides (smaller instances, fewer resources)
│   ├── staging.tfvars       # Staging overrides
│   └── prod.tfvars          # Production overrides
│
└── modules/
    ├── s3/                  # S3 bucket (versioning, encryption, lifecycle)
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── ecr/                 # ECR repository (image lifecycle policy)
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── batch/               # AWS Batch compute environment + job definitions
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── ecs/                 # ECS Fargate cluster + service + ALB
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── iam/                 # IAM roles and policies
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    ├── networking/          # VPC, subnets, security groups
    │   ├── main.tf
    │   ├── variables.tf
    │   └── outputs.tf
    └── monitoring/          # CloudWatch log groups, alarms, SNS topics
        ├── main.tf
        ├── variables.tf
        └── outputs.tf
```

### 7.2 Multiple Environments

| Property         | dev                      | staging                   | prod                      |
|------------------|--------------------------|---------------------------|---------------------------|
| S3 bucket        | `eye-hear-u-dev-data`    | `eye-hear-u-staging-data` | `eye-hear-u-prod-data`    |
| Batch vCPUs      | 2                        | 4                         | 8                         |
| ECS task CPU     | 256 (0.25 vCPU)         | 512 (0.5 vCPU)           | 1024 (1 vCPU)            |
| ECS task memory  | 512 MB                   | 1024 MB                   | 2048 MB                   |
| SageMaker inst.  | `ml.t3.medium` (CPU)     | `ml.g4dn.xlarge` (GPU)   | `ml.g4dn.xlarge` (GPU)   |
| CloudWatch retain| 7 days                   | 30 days                   | 90 days                   |
| S3 versioning    | Enabled                  | Enabled                   | Enabled                   |
| SNS alerts       | Email only               | Email + Slack             | Email + Slack + PagerDuty |
| DynamoDB         | On-demand (pay-per-req)  | On-demand                 | On-demand                 |

Environments are selected via Terraform variable files:
```bash
# Deploy dev
terraform apply -var-file=environments/dev.tfvars

# Deploy prod
terraform apply -var-file=environments/prod.tfvars
```

---

## 8. Pipeline Orchestration (Apache Airflow)

Apache Airflow orchestrates the pipeline stages as a Directed Acyclic Graph (DAG). In development we run Airflow locally via Docker Compose; in production we can use **Amazon MWAA** (Managed Workflows for Apache Airflow) or a self-hosted Airflow on ECS.

### 8.1 DAG Definition

```
eye_hear_u_full_pipeline (DAG)
│
├── ingest_asl_citizen  ──┐
├── ingest_wlasl        ──┼──→  preprocess_clips  ──→  build_dataset  ──→  validate
└── ingest_msasl        ──┘
```

- **Ingestion tasks** run in parallel (no dependencies between sources).
- **Preprocessing** waits for all ingestion tasks to complete.
- **Build** waits for preprocessing.
- **Validate** runs last and publishes a pass/fail metric to CloudWatch.

### 8.2 Task Types

Each Airflow task submits a job to **AWS Batch** (in cloud environments) or runs a local Docker container (in dev):

| Task                  | Container         | AWS Batch Job Queue     | Timeout  |
|-----------------------|-------------------|-------------------------|----------|
| `ingest_asl_citizen`  | `data-pipeline`   | `eye-hear-u-{env}-queue`| 30 min   |
| `ingest_wlasl`        | `data-pipeline`   | `eye-hear-u-{env}-queue`| 30 min   |
| `ingest_msasl`        | `data-pipeline`   | `eye-hear-u-{env}-queue`| 30 min   |
| `preprocess_clips`    | `data-pipeline`   | `eye-hear-u-{env}-queue`| 4 hours  |
| `build_dataset`       | `data-pipeline`   | `eye-hear-u-{env}-queue`| 15 min   |
| `validate`            | `data-pipeline`   | `eye-hear-u-{env}-queue`| 15 min   |

### 8.3 Scheduling

| Trigger         | Use Case                                                      |
|-----------------|---------------------------------------------------------------|
| Manual          | First-time full pipeline run                                  |
| On S3 upload    | Re-ingest a source when new raw videos are added to S3        |
| Weekly (cron)   | Re-validate dataset integrity (catch silent corruption)       |
| Post-training   | After a SageMaker training job completes, validate the model  |

---

## 9. Pipeline Code

All source code lives in `data/scripts/`:

| File                        | Stage         | Description                                                      |
|-----------------------------|---------------|------------------------------------------------------------------|
| `pipeline_config.py`        | Shared config | Paths (local + S3), URLs, preprocessing constants, environment   |
| `ingest_asl_citizen.py`     | Ingestion     | Parse ASL Citizen metadata, validate, write normalised CSV       |
| `ingest_wlasl.py`           | Ingestion     | Download WLASL JSON, parse instances, validate, write CSV        |
| `ingest_msasl.py`           | Ingestion     | Download MS-ASL JSONs, parse splits, validate, write CSV         |
| `preprocess_clips.py`       | Preprocessing | Trim → sample 16 frames → resize 224×224 → write .mp4           |
| `build_unified_dataset.py`  | Building      | Filter rare glosses, create label_map.json, dataset_stats.json   |
| `validate.py`               | Validation    | File existence, frame/resolution checks, signer-leak detection   |
| `requirements.txt`          | Dependencies  | opencv-python, numpy, requests, boto3                            |

---

## 10. Next Steps (Features Not Yet Implemented)

### 10.1 Automated WLASL & MS-ASL Video Downloading

Currently, ingestion scripts assume that raw video files have already been uploaded to S3 (or placed locally). A future enhancement is to automate video downloading:
- **WLASL / MS-ASL:** Videos originate from YouTube. We plan to integrate `yt-dlp` to download clips by URL and trim to the annotated temporal boundaries automatically.
- **Handling dead links:** Both WLASL and MS-ASL contain a significant proportion of broken YouTube URLs. The pipeline should log failures and report a coverage summary.

### 10.2 Gloss Vocabulary Alignment Across Datasets

Different datasets may use slightly different gloss labels for the same sign (e.g., `"bathroom"` vs. `"restroom"`). A future step is to build a synonym mapping table so that supplementary clips from WLASL and MS-ASL are correctly matched to the ASL Citizen gloss vocabulary.

### 10.3 DVC Integration for Dataset Versioning

[DVC (Data Version Control)](https://dvc.org/) would track large data files alongside Git, with S3 as the remote backend:
```bash
dvc init
dvc remote add -d s3remote s3://eye-hear-u-prod-data/dvc/
dvc add data/processed/
git add data/processed/.dvc
```
This gives the team reproducible dataset versions tied to Git commits.

### 10.4 Negative Example Pipeline

As described in Part One, a negative example dataset is important for reducing false positives:
- Ingest non-ASL gesture videos (from datasets like Jester or Cambridge Hand Gesture).
- Add a special `"__negative__"` class to the label map.
- Include negative clips in the training set.

### 10.5 CI/CD Validation

GitHub Actions workflow that:
- Runs `validate.py` on every pull request to `main`.
- Builds and pushes the `data-pipeline` Docker image to ECR on merge.
- Runs `terraform plan` on infrastructure changes (no auto-apply).

### 10.6 Frame-Level Feature Caching

Cache per-frame CNN features (from a frozen backbone) as `.npy` arrays in S3 to eliminate redundant forward passes during fine-tuning, reducing epoch time by approximately 3–5×.
