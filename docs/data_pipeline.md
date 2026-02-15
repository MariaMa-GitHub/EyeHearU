# Data Processing Pipeline — Eye Hear U

**Version:** 2026-02-14  
**Authors:** CSC490 Team  
**Status:** Initial implementation

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Detailed Pipeline Diagram with AWS Services](#3-detailed-pipeline-diagram-with-aws-services)
4. [Data Schemas](#4-data-schemas)
5. [Pipeline Stages in Detail](#5-pipeline-stages-in-detail)
6. [When Pipelines Run & Use Cases](#6-when-pipelines-run--use-cases)
7. [Code Implementation](#7-code-implementation)
8. [Technology Stack](#8-technology-stack)
9. [Next Steps & Future Enhancements](#9-next-steps--future-enhancements)

---

## 1. Pipeline Overview

Eye Hear U is a real-time ASL-to-English translation system. The data processing
pipeline transforms raw ASL video datasets into clean, labeled image datasets
suitable for training a CNN+Transformer classifier.

### Data Sources

| Source       | Type     | Size    | Description                                  |
|-------------|----------|---------|----------------------------------------------|
| ASL Citizen | Videos   | ~84K    | Crowdsourced, 2,731 signs, 52 signers (Microsoft Research) |
| WLASL       | Videos   | ~21K    | YouTube-sourced, 2,000 signs (academic)      |
| Custom      | Videos   | TBD     | Team-recorded samples for gap-filling        |
| User App    | Images   | Growing | Real-time predictions from the mobile app    |

### Pipeline Summary

```
 Raw Video Data ──▶ Frame Extraction ──▶ Cleaning ──▶ Transformation ──▶ Split ──▶ Model Training
  (S3 raw/)          (Lambda)           (Glue)       (Lambda/Batch)    (Glue)     (SageMaker)
```

---

## 2. High-Level Architecture Diagram

This diagram maps directly to the restaurant-review example's high-level
architecture, adapted for ASL sign language recognition.

```
                                    ┌─────────────────────┐
                                    │   Internal ML        │
                                    │   Dashboard          │
                                    │   (Amazon QuickSight)│
                                    └────────▲────────────┘
                                             │
┌──────────┐    ┌──────────────┐    ┌────────┴────────────┐    ┌─────────────────┐
│ Mobile   │    │   API        │    │  SQL Data Store      │    │  Analytics Data  │
│ App      │───▶│   Gateway    │───▶│  (Aurora PostgreSQL) │    │  Store           │
│ (React   │    │   (Amazon    │    │                      │    │  (Amazon         │
│  Native) │    │    API GW)   │    │  sign_glosses        │    │   Redshift)      │
└──────────┘    └──────┬───────┘    │  signers             │    └────────▲────────┘
                       │            │  sign_videos         │             │
                       ▼            │  extracted_frames    │    ┌────────┴────────┐
                ┌──────────────┐    │  processed_images    │    │  ETL Layer       │
                │ CRUD Service │    │  label_map           │    │  (AWS Glue)      │
                │ Layer        │───▶│  pipeline_runs       │    │                  │
                │ (AWS Lambda  │    └──────────────────────┘    │  Catalog +       │
                │  + Fargate)  │                                │  Transform +     │
                └──────┬───────┘    ┌──────────────────────┐    │  Quality checks  │
                       │            │  Document Data Store  │    └────────▲────────┘
                       ├───────────▶│  (Firebase Firestore  │             │
                       │            │   + DynamoDB)         │    ┌────────┴────────┐
                       │            │                       │    │  Streaming Layer │
                       │            │  translations         │───▶│  (Amazon Kinesis)│
                       │            │  sessions             │    │                  │
                       │            │  model_registry       │    │  Real-time       │
                       │            │  vocabulary           │    │  prediction logs │
                       │            └──────────────────────┘    └─────────────────┘
                       │
                       │            ┌──────────────────────┐
                       │            │  ML Analytics Layer   │
                       ▼            │                       │
                ┌──────────────┐    │  Confusion analysis   │
                │ Model Service│───▶│  Per-class accuracy   │
                │ Layer        │    │  Confidence calibration│
                │ (SageMaker   │    │  Data drift detection │
                │  Endpoint)   │    └──────────────────────┘
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐    ┌──────────────────────┐
                │ Model        │    │  Object Store         │
                │ Training     │◀──▶│  (Amazon S3)          │
                │ Layer        │    │                       │
                │ (SageMaker   │    │  s3://eyehearu-data-  │
                │  Training)   │    │    lake/raw/           │
                └──────────────┘    │  s3://eyehearu-data-  │
                                    │    lake/processed/     │
                                    │  s3://eyehearu-       │
                                    │    models/             │
                                    └──────────────────────┘

Legend:
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │User Layer│ │API Layer │ │Data Layer│ │ ML Layer │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

---

## 3. Detailed Pipeline Diagram with AWS Services

### 3.1 Offline Training Pipeline (Batch)

This pipeline runs **on-demand** when new data is available or model retraining
is needed.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          OFFLINE TRAINING PIPELINE                                       │
│                                                                                         │
│  ┌───────────────┐    ┌────────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│  │  DATA SOURCES  │    │  S3 RAW ZONE   │    │  AWS LAMBDA    │    │  S3 CLEANED ZONE │  │
│  │               │    │  (Data Lake)   │    │  (Frame        │    │                  │  │
│  │ ASL Citizen   │───▶│               │───▶│   Extraction)  │───▶│  Validated       │  │
│  │  (84K videos) │    │ raw/asl_citizen│    │               │    │  frames only     │  │
│  │               │    │               │    │ • Open video   │    │                  │  │
│  │ WLASL         │───▶│ raw/wlasl/    │    │ • Extract 5    │    │ cleaned/frames/  │  │
│  │  (21K videos) │    │               │    │   frames/video │    │                  │  │
│  │               │    │ raw/custom/   │    │ • Validate     │    └────────┬─────────┘  │
│  │ Custom        │───▶│               │    │   dimensions   │             │             │
│  │  recordings   │    └───────┬────────┘    └────────────────┘             │             │
│  └───────────────┘            │                                           │             │
│                               │                                           ▼             │
│                    ┌──────────▼────────┐                       ┌──────────────────┐     │
│                    │  AWS GLUE         │                       │  AWS LAMBDA      │     │
│                    │  (Data Catalog)   │                       │  (Cleaning)      │     │
│                    │                   │                       │                  │     │
│                    │ • Crawl new data  │                       │ • Remove black   │     │
│                    │ • Update catalog  │                       │   frames         │     │
│                    │ • Schema detection│                       │ • Remove blurry  │     │
│                    │ • Partition by    │                       │   images         │     │
│                    │   dataset/gloss   │                       │ • Deduplicate    │     │
│                    └──────────────────┘                       │   (perceptual    │     │
│                                                               │    hash)         │     │
│                                                               │ • Log quality    │     │
│                                                               │   metrics        │     │
│                                                               └────────┬─────────┘     │
│                                                                        │               │
│                                                                        ▼               │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐  ┌──────────────────┐  │
│  │  SAGEMAKER     │    │  S3 MODELS     │    │  AWS BATCH     │  │  S3 PROCESSED    │  │
│  │  (Training)    │    │  (Checkpoint   │    │  (Transform)   │  │  ZONE            │  │
│  │                │    │   Store)       │    │                │  │  (Data Warehouse)│  │
│  │ CNN+Transformer│◀──▶│               │    │ • Resize 224²  │◀─│                  │  │
│  │ ASLClassifier  │    │ best_model.pt │    │ • Hand crop    │  │ processed/images/│  │
│  │                │    │ label_map.json│    │   (MediaPipe)  │  │   train/hello/   │  │
│  │ • 30 epochs   │    │ config.json   │    │ • Normalize    │  │   val/hello/     │  │
│  │ • AdamW       │    │               │    │ • Split 80/10/ │  │   test/hello/    │  │
│  │ • Cosine LR   │    └────────────────┘    │   10           │  │                  │  │
│  │ • Early stop  │                          └────────────────┘  │ label_map.json   │  │
│  └───────┬────────┘                                             │ dataset_stats.json│  │
│          │                                                      └──────────────────┘  │
│          ▼                                                                             │
│  ┌────────────────┐    ┌────────────────┐                                              │
│  │  SAGEMAKER     │    │  DYNAMODB      │                                              │
│  │  (Evaluation)  │───▶│  (Model        │                                              │
│  │                │    │   Registry)    │                                              │
│  │ • Accuracy     │    │                │                                              │
│  │ • Top-5 acc    │    │ model_version  │                                              │
│  │ • Per-class    │    │ accuracy       │                                              │
│  │ • Confusion    │    │ dataset_version│                                              │
│  └────────────────┘    │ deployed_at    │                                              │
│                        └────────────────┘                                              │
│                                                                                        │
│  Orchestration: AWS Step Functions                                                     │
│  Monitoring:    Amazon CloudWatch                                                      │
│  Alerts:        Amazon SNS                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Online Inference Pipeline (Real-time)

This pipeline runs **on every prediction request** from the mobile app.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          ONLINE INFERENCE PIPELINE                                │
│                                                                                  │
│  ┌──────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │  Mobile  │    │  Amazon    │    │  AWS Lambda │    │  SAGEMAKER ENDPOINT  │   │
│  │  App     │───▶│  API       │───▶│  / Fargate  │───▶│  (Model Inference)   │   │
│  │  (iOS)   │    │  Gateway   │    │  (FastAPI)  │    │                      │   │
│  │          │    │            │    │             │    │  1. Preprocess image  │   │
│  │ Camera   │    │ POST       │    │ Validate    │    │     (224x224, norm)  │   │
│  │ Capture  │    │ /predict   │    │ file type   │    │  2. CNN backbone     │   │
│  │          │◀───│            │◀───│ and size    │◀───│  3. Transformer enc  │   │
│  │ Display  │    │ JSON       │    │             │    │  4. Classify (62 cls)│   │
│  │ + TTS    │    │ response   │    │ Format resp │    │  5. Top-K softmax    │   │
│  └──────────┘    └────────────┘    └──────┬──────┘    └──────────────────────┘   │
│                                           │                                      │
│                                           ▼                                      │
│                                   ┌───────────────┐    ┌──────────────────────┐   │
│                                   │  FIRESTORE    │    │  AMAZON KINESIS      │   │
│                                   │  (Log         │    │  (Stream predictions │   │
│                                   │   prediction) │───▶│   to analytics)      │   │
│                                   │               │    │                      │   │
│                                   │ translations  │    │  → Redshift          │   │
│                                   │ sessions      │    │  → QuickSight        │   │
│                                   └───────────────┘    └──────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Analytics Pipeline (Batch + Streaming)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          ANALYTICS PIPELINE                                      │
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  Firestore   │    │  Amazon      │    │  AWS Glue    │    │  Amazon      │   │
│  │  (prediction │───▶│  Kinesis     │───▶│  (ETL to     │───▶│  Redshift    │   │
│  │   logs)      │    │  Data        │    │   analytics  │    │  (Analytics  │   │
│  │              │    │  Firehose    │    │   warehouse) │    │   DW)        │   │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘   │
│                                                                     │           │
│                                                              ┌──────▼───────┐   │
│  Metrics tracked:                                            │  Amazon      │   │
│  • Predictions per day/hour                                  │  QuickSight  │   │
│  • Accuracy by sign category (greeting, medical, etc.)       │  (Dashboard) │   │
│  • Confidence distribution                                   │              │   │
│  • Most common signs predicted                               │  Internal BI │   │
│  • User feedback rates (correct/wrong)                       │  for ML team │   │
│  • Model latency percentiles (p50, p95, p99)                 └──────────────┘   │
│  • Data drift detection (input distribution shift)                              │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Schemas

Full data schemas are documented in [`data_schema.md`](./data_schema.md).

### Summary of Key Entities

| Entity            | Store                  | Purpose                              |
|-------------------|------------------------|--------------------------------------|
| `sign_glosses`    | Aurora PostgreSQL      | Master vocabulary (62 signs)         |
| `signers`         | Aurora PostgreSQL      | Signer metadata from datasets        |
| `sign_videos`     | Aurora PostgreSQL      | Raw video metadata + S3 paths        |
| `extracted_frames`| Aurora PostgreSQL      | Per-frame metadata after extraction  |
| `processed_images`| Aurora PostgreSQL      | Final training-ready image metadata  |
| `label_map`       | Aurora PostgreSQL      | Versioned gloss→index mapping        |
| `pipeline_runs`   | Aurora PostgreSQL      | Pipeline execution audit trail       |
| `translations`    | Firestore              | App prediction logs                  |
| `sessions`        | Firestore              | User session tracking                |
| `model_registry`  | DynamoDB               | Deployed model versions              |
| Raw videos        | S3 (raw zone)          | Binary video files                   |
| Processed images  | S3 (processed zone)    | 224x224 normalized JPEGs             |
| Model checkpoints | S3 (models bucket)     | .pt files + configs                  |

---

## 5. Pipeline Stages in Detail

### Stage 1: Data Ingestion

**Purpose:** Download raw datasets and store in the data lake (S3 raw zone).

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | ASL Citizen ZIP (Microsoft), WLASL JSON + videos     |
| **Output**      | Videos in `s3://eyehearu-data-lake/raw/`             |
| **AWS Services**| S3, Lambda (trigger), Glue Crawler                   |
| **Local Code**  | `data/scripts/download_asl_citizen.py`, `download_wlasl.py` |
| **Trigger**     | Manual (new dataset release) or scheduled monthly    |

**Steps:**
1. Download ASL Citizen ZIP from Microsoft (wget/requests)
2. Extract videos, parse metadata CSV
3. Download WLASL metadata JSON, identify available videos
4. Upload to S3 raw zone with dataset/gloss partitioning
5. Run Glue Crawler to update Data Catalog

### Stage 2: Data Cleaning

**Purpose:** Validate raw data, remove corrupted/unusable files, deduplicate.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Raw videos/frames from S3 raw zone                   |
| **Output**      | Validated frames in `s3://eyehearu-data-lake/cleaned/`|
| **AWS Services**| Lambda (per-file validation), Glue (batch cleaning)  |
| **Local Code**  | `data/scripts/pipeline.py` → `stage_clean()`         |
| **Trigger**     | After ingestion stage completes                      |

**Cleaning Rules:**
- Remove videos < 0.3 seconds duration
- Remove frames < 64x64 pixels
- Remove black frames (mean pixel < 10)
- Remove blurry frames (Laplacian variance < 10)
- Deduplicate using perceptual hashing (average hash)
- Flag classes with < 5 samples as "needs supplementation"

### Stage 3: Data Transformation

**Purpose:** Transform cleaned images into model-ready format.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Cleaned frames from S3 cleaned zone                  |
| **Output**      | 224x224 normalized images in S3 processed zone       |
| **AWS Services**| AWS Batch (heavy compute), Lambda (lightweight)      |
| **Local Code**  | `data/scripts/pipeline.py` → `stage_transform()`     |
| **Trigger**     | After cleaning stage completes                       |

**Transformation Steps:**
1. **Hand Detection** — MediaPipe Hands detects hand region bounding box
2. **Hand Cropping** — Crop to hand region + 25% padding (if detected)
3. **Resize** — Resize to 224x224 using INTER_AREA interpolation
4. **Quality Save** — Save as JPEG quality=95

*Note: Pixel normalization (ImageNet mean/std) and data augmentation are applied
at training time in `ml/training/dataset.py`, not in the preprocessing pipeline.*

### Stage 4: Data Loading

**Purpose:** Split processed data into train/val/test and generate metadata.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Processed images from S3 processed zone              |
| **Output**      | train/val/test splits + label_map + stats            |
| **AWS Services**| Glue (partition management), S3 (organized storage)  |
| **Local Code**  | `data/scripts/pipeline.py` → `stage_load()`          |
| **Trigger**     | After transformation stage completes                 |

**Split Ratios:** 80% train / 10% val / 10% test (stratified by class)

### Stage 5: Reporting

**Purpose:** Generate pipeline execution report and data quality metrics.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Metrics from all previous stages                     |
| **Output**      | JSON report in `data/metadata/`                      |
| **AWS Services**| CloudWatch (metrics), QuickSight (dashboard), SNS    |
| **Local Code**  | `data/scripts/pipeline.py` → `stage_report()`        |
| **Trigger**     | After all stages complete                            |

---

## 6. When Pipelines Run & Use Cases

### Pipeline Schedule

| Pipeline              | Trigger                     | Frequency           | Use Case                        |
|-----------------------|-----------------------------|---------------------|--------------------------------|
| **Full Training**     | New dataset available       | On-demand (~monthly)| Initial model training          |
| **Incremental Ingest**| New custom recordings       | Weekly              | Add team-recorded samples       |
| **Retraining**        | Data drift detected         | Monthly/on-demand   | Model performance degradation   |
| **Analytics ETL**     | Prediction logs accumulate  | Daily (batch)       | Usage dashboards & reporting    |
| **Real-time Inference**| User taps "Capture"        | Per-request         | Core app functionality          |

### Use Case: Initial Model Training

```
Trigger: First-time setup / new target vocabulary defined
Schedule: One-time, then on-demand

1. Download ASL Citizen dataset (Stage 1: Ingest)
2. Download WLASL dataset (Stage 1: Ingest)
3. Extract frames from all target-vocab videos (Stage 2: Clean)
4. Validate and deduplicate frames (Stage 2: Clean)
5. Resize, hand-crop, normalize (Stage 3: Transform)
6. Split 80/10/10 (Stage 4: Load)
7. Train ASLClassifier on SageMaker (Stage 5: Train)
8. Evaluate and register best model (Stage 5: Report)
```

### Use Case: Restaurant Scenario Enhancement

```
Trigger: Team records additional restaurant-specific signs
Schedule: As needed

1. Record new videos for: eat, drink, hot, cold, more, enough, check
2. Ingest recordings to S3 (Stage 1)
3. Run cleaning + transformation (Stages 2-3)
4. Merge with existing processed data
5. Re-split to maintain balanced splits (Stage 4)
6. Fine-tune existing model on new data (only unfreezing last layers)
```

### Use Case: Model Performance Monitoring

```
Trigger: Automated CloudWatch alarm (accuracy drops below threshold)
Schedule: Continuous monitoring, retraining triggered automatically

1. Kinesis streams prediction logs to Redshift
2. Glue ETL computes accuracy metrics per sign category
3. If accuracy < 80% for any category → trigger SNS alert
4. ML team reviews confusion matrix and data distribution
5. If data drift detected → trigger retraining pipeline
```

---

## 7. Code Implementation

### Project Structure

```
data/
├── scripts/
│   ├── download_asl_citizen.py   # ASL Citizen dataset downloader
│   ├── download_wlasl.py         # WLASL dataset downloader
│   ├── preprocess.py             # Image preprocessing utilities
│   └── pipeline.py               # Main pipeline orchestrator
├── raw/                          # Raw downloaded data (gitignored)
│   ├── asl_citizen/
│   └── wlasl/
├── processed/                    # Processed images + metadata (gitignored)
│   ├── images/
│   │   ├── all/                  # Before splitting
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── label_map.json
│   └── dataset_stats.json
├── metadata/                     # Pipeline metadata + provenance
│   ├── raw_data_catalog.json
│   ├── dataset_manifest.json
│   ├── asl_citizen_quality_report.json
│   ├── asl_citizen_provenance.csv
│   └── pipeline_run_*.json
└── logs/                         # Pipeline execution logs
    └── pipeline_*.log
```

### Running the Pipeline

```bash
# Full pipeline (all stages)
cd data/scripts
python pipeline.py --stage all

# Individual stages
python pipeline.py --stage ingest       # Download datasets
python pipeline.py --stage clean        # Validate & clean
python pipeline.py --stage transform    # Resize, crop, normalize
python pipeline.py --stage load         # Split & generate metadata
python pipeline.py --stage report       # Generate reports

# Download ASL Citizen dataset specifically
python download_asl_citizen.py

# Download WLASL dataset specifically
python download_wlasl.py
```

### Key Code Files

| File                        | Purpose                                  | Lines |
|-----------------------------|------------------------------------------|-------|
| `pipeline.py`               | Main orchestrator with 5 stages          | ~500  |
| `download_asl_citizen.py`   | ASL Citizen download + frame extraction  | ~300  |
| `download_wlasl.py`         | WLASL download + frame extraction        | ~180  |
| `preprocess.py`             | MediaPipe hand cropping + splitting      | ~160  |

---

## 8. Technology Stack

### AWS Services (Production)

| Layer                | Service                  | Purpose                           |
|----------------------|--------------------------|-----------------------------------|
| **Compute**          | AWS Lambda               | Frame extraction, validation      |
|                      | AWS Batch                | Heavy image transformation        |
|                      | Amazon SageMaker         | Model training + inference        |
|                      | AWS Fargate (ECS)        | Backend API hosting               |
| **Storage**          | Amazon S3                | Data lake + model checkpoints     |
|                      | Amazon Aurora PostgreSQL | Structured metadata (pipeline)    |
|                      | Amazon DynamoDB          | Model registry, pipeline state    |
|                      | Firebase Firestore       | App-facing data (translations)    |
| **Orchestration**    | AWS Step Functions       | Pipeline workflow orchestration   |
|                      | Amazon EventBridge       | Scheduled triggers                |
| **Streaming**        | Amazon Kinesis Firehose  | Real-time prediction log ingestion|
| **Analytics**        | Amazon Redshift          | Analytics data warehouse          |
|                      | Amazon QuickSight        | BI dashboards                     |
|                      | AWS Glue                 | ETL + Data Catalog                |
| **Networking**       | Amazon API Gateway       | REST API endpoint                 |
|                      | Amazon CloudFront        | CDN (future: reference videos)    |
| **Monitoring**       | Amazon CloudWatch        | Logs, metrics, alarms             |
|                      | Amazon SNS               | Alert notifications               |
| **CI/CD**            | Amazon ECR               | Container registry                |
|                      | AWS CodePipeline         | Deployment automation             |

### Open-Source Frameworks (Local Development)

| Tool          | Version | Purpose                                    |
|---------------|---------|---------------------------------------------|
| Python        | 3.11+   | Pipeline scripting language                 |
| OpenCV        | 4.x     | Video processing, frame extraction          |
| MediaPipe     | 0.10+   | Hand detection and region cropping          |
| PyTorch       | 2.2+    | Model training and inference                |
| Torchvision   | 0.17+   | Image transforms, pretrained backbones      |
| Pandas        | 2.x     | Metadata handling, provenance tracking      |
| NumPy         | 1.26+   | Numerical operations                        |
| FastAPI       | 0.110+  | Backend REST API                            |
| scikit-learn  | 1.4+    | Data splitting, evaluation metrics          |
| tqdm          | 4.x     | Progress bars for pipeline stages           |

---

## 9. Next Steps & Future Enhancements

### Implemented (v1.0)
- [x] ASL Citizen dataset download and frame extraction
- [x] WLASL dataset metadata parsing and filtering
- [x] Video validation (duration, resolution, corruption checks)
- [x] Frame cleaning (black frame removal, blur detection, deduplication)
- [x] Hand region detection and cropping (MediaPipe)
- [x] Image resizing and normalization
- [x] Train/val/test splitting (80/10/10)
- [x] Label map generation
- [x] Pipeline orchestration with stage-by-stage execution
- [x] Data quality reporting and provenance tracking
- [x] Comprehensive data schemas (SQL, Document, Object Store)

### Not Yet Implemented (Future Enhancements)

#### Short-Term (Next Sprint)
- [ ] **AWS S3 Integration** — Upload raw/processed data to S3 buckets instead of local storage
- [ ] **AWS Step Functions** — Replace local orchestrator with cloud-native workflow
- [ ] **Data Augmentation Pipeline** — Add rotation, color jitter, scale variations as a separate stage
- [ ] **Automated Retraining Trigger** — CloudWatch alarm on accuracy drop → auto-retrain

#### Medium-Term (Next Month)
- [ ] **Kinesis Streaming** — Stream real-time predictions to analytics warehouse
- [ ] **Redshift Analytics** — Build prediction analytics warehouse for dashboards
- [ ] **QuickSight Dashboard** — Visual monitoring of model performance and usage
- [ ] **Video-Level Processing** — Process video sequences instead of single frames (for future temporal models like I3D or SlowFast)
- [ ] **Signer-Stratified Splitting** — Ensure no signer appears in both train and test sets to prevent identity leakage

#### Long-Term (Future Semester)
- [ ] **Continuous Learning Pipeline** — Incorporate user feedback (correct/wrong) to improve model over time
- [ ] **Federated Data Collection** — Allow users to opt-in contributing sign videos for underrepresented classes
- [ ] **Multi-Dataset Fusion** — Automated pipeline to merge ASL Citizen + WLASL + MS-ASL + custom data with conflict resolution
- [ ] **Real-time Video Classification** — Switch from single-frame to video clip classification using temporal models
- [ ] **Data Versioning** — Integrate DVC (Data Version Control) for reproducible dataset versions
- [ ] **Cost Optimization** — S3 Intelligent-Tiering, Spot Instances for SageMaker training
- [ ] **Data Privacy Compliance** — Add PII detection and anonymization for any user-contributed data
