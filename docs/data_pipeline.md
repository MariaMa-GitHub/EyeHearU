# Data Processing Pipeline — Eye Hear U

**Version:** 2026-02-14  
**Authors:** CSC490 Team  
**Status:** Initial implementation

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Detailed Data Processing Pipeline](#3-detailed-data-processing-pipeline)
4. [Data Schemas](#4-data-schemas)
5. [Pipeline Stages in Detail](#5-pipeline-stages-in-detail)
6. [When Pipelines Run & Use Cases](#6-when-pipelines-run--use-cases)
7. [Code Implementation](#7-code-implementation)
8. [Technology Stack](#8-technology-stack)
9. [Next Steps & Future Enhancements](#9-next-steps--future-enhancements)

---

## 1. Pipeline Overview

Eye Hear U is a real-time ASL-to-English translation system. The data processing
pipeline transforms raw ASL video datasets into pose-keypoint sequences suitable
for training an **ST-GCN (Spatial-Temporal Graph Convolutional Network)**
classifier. Videos are processed as temporal sequences — not cut into individual
frames or images.

### Data Sources

| Source       | Type     | Size    | Description                                  |
|-------------|----------|---------|----------------------------------------------|
| ASL Citizen | Videos   | ~84K    | Crowdsourced, 2,731 signs, 52 signers (Microsoft Research) |
| WLASL       | Videos   | ~21K    | YouTube-sourced, 2,000 signs (academic)      |
| MS-ASL      | Videos   | ~25K    | YouTube-sourced, 1,000 signs, 222 signers (Microsoft Research) |
| User App    | Video clips | Growing | Real-time predictions from the mobile app |

### Pipeline Summary

```
 ASL Citizen ─┐                                                  ┌─▶ train.csv (80%)
 WLASL ───────┼──▶ Combine ──▶ Validate ──▶ Pose Extract ──▶ Split ──▶ val.csv   (10%)
 MS-ASL ──────┘    metadata     videos       → .npy files          └─▶ test.csv  (10%)
                                                                        + label_map.json
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
                       ▼            │  extracted_poses     │    ┌────────┴────────┐
                ┌──────────────┐    │  label_map           │    │  ETL Layer       │
                │ CRUD Service │    │  pipeline_runs       │    │  (AWS Glue)      │
                │ Layer        │───▶│                      │    │                  │
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

## 3. Detailed Data Processing Pipeline

This diagram shows the complete data processing flow implemented in
`pipeline.py`.  ASL Citizen, WLASL, and MS-ASL are ingested, validated,
and combined into **one unified dataset** of pose-keypoint sequences for
ST-GCN training.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      DATA PROCESSING PIPELINE  (pipeline.py)                         │
│                                                                                     │
│  STAGE 1 ─ INGEST                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐                │
│  │  ASL Citizen       │  │  WLASL            │  │  MS-ASL           │                │
│  │  (~84K videos)     │  │  (~21K videos)    │  │  (~25K videos)    │                │
│  │                   │  │                   │  │                   │                │
│  │ metadata.csv      │  │ WLASL_v0.3.json   │  │ MSASL_train.json  │                │
│  │ (user, file,      │  │ (gloss →          │  │ MSASL_val.json    │                │
│  │  gloss)           │  │  instances[])     │  │ MSASL_test.json   │                │
│  └────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘                │
│           │                      │                      │                           │
│           └──────────────┬───────┴──────────────────────┘                           │
│                          ▼                                                          │
│            ┌───────────────────────────┐                                            │
│            │  BUILD COMBINED METADATA   │                                            │
│            │                           │                                            │
│            │ • Parse ASL Citizen CSV   │     ┌─────────────────────────────┐         │
│            │ • Convert WLASL JSON →    │────▶│  processed/                 │         │
│            │   same CSV schema         │     │    combined_metadata.csv    │         │
│            │ • Convert MS-ASL JSONs →  │     │                             │         │
│            │   same CSV schema         │     │  user,file,gloss,dataset    │         │
│            │ • Normalise all into      │     │  (one row per video)        │         │
│            │   (user, file, gloss,     │     │                             │         │
│            │    dataset) rows          │     │  ~130K videos total         │         │
│            └───────────────────────────┘     └──────────────┬──────────────┘         │
│                                                                 │                    │
│  STAGE 2 ─ CLEAN                                                │                    │
│                ┌───────────────────────────┐                    │                    │
│                │  VIDEO VALIDATION          │◀───────────────────┘                    │
│                │                           │                                        │
│                │  For every .mp4 across    │                                        │
│                │  all raw/ subdirectories: │     ┌─────────────────────────────┐     │
│                │                           │────▶│  metadata/                  │     │
│                │  ✓ Codec decodable?       │     │    validated_video_catalog  │     │
│                │  ✓ Duration ≥ 0.3s?       │     │    .json                    │     │
│                │  ✓ Resolution ≥ 64×64?    │     │                             │     │
│                │  ✓ Class balance analysis │     │  + quality_issues[]         │     │
│                └───────────────────────────┘     └─────────────────────────────┘     │
│                                                                                     │
│  STAGE 3 ─ POSE EXTRACTION                                                          │
│                ┌───────────────────────────┐                                        │
│                │  MEDIAPIPE HOLISTIC        │                                        │
│                │                           │                                        │
│                │  For each video in        │                                        │
│                │  combined_metadata.csv:   │     ┌─────────────────────────────┐     │
│                │                           │     │  processed/poses/           │     │
│                │  • Read all frames        │────▶│    asl_citizen_*.npy        │     │
│                │  • Extract 543 keypoints  │     │    wlasl_*.npy              │     │
│                │    per frame:             │     │    ms_asl_*.npy             │     │
│                │    33 pose + 21 R hand    │     │                             │     │
│                │    + 21 L hand + 468 face │     │  Shape: (T, 543, 2)        │     │
│                │  • Save .npy (T,543,2)    │     │  T = frames in that video  │     │
│                │                           │     ├─────────────────────────────┤     │
│                │  Skip if .npy exists      │     │  processed/                 │     │
│                │  (incremental)            │────▶│    pose_mapping.csv         │     │
│                └───────────────────────────┘     │  (video_filename → npy)     │     │
│                                                  └─────────────────────────────┘     │
│                                                                                     │
│  STAGE 4 ─ SPLIT                                                                    │
│                ┌───────────────────────────┐                                        │
│                │  STRATIFIED SPLIT          │                                        │
│                │                           │                                        │
│                │  Load combined_metadata   │     ┌─────────────────────────────┐     │
│                │  .csv (all datasets)      │     │  processed/                 │     │
│                │                           │     │    train.csv  (80%)         │     │
│                │  • Group by gloss         │────▶│    val.csv    (10%)         │     │
│                │  • 80 / 10 / 10 split     │     │    test.csv   (10%)         │     │
│                │    per class              │     │                             │     │
│                │  • Write split CSVs       │     │  Same schema:              │     │
│                │    (user,file,gloss,      │     │  user,file,gloss,dataset   │     │
│                │     dataset)              │     ├─────────────────────────────┤     │
│                │                           │     │  processed/                 │     │
│                │  • Build label_map.json   │────▶│    label_map.json           │     │
│                │    (gloss → int index)    │     │    dataset_stats.json       │     │
│                └───────────────────────────┘     └─────────────────────────────┘     │
│                                                                                     │
│  STAGE 5 ─ REPORT                                                                   │
│                ┌───────────────────────────┐     ┌─────────────────────────────┐     │
│                │  PIPELINE REPORT           │────▶│  metadata/                  │     │
│                │                           │     │    pipeline_report_*.json   │     │
│                │  • Pose file count + size │     │    pipeline_run_*.json      │     │
│                │  • Per-split video counts │     │                             │     │
│                │  • Class counts per split │     │  Logged to console +        │     │
│                │  • Error summary          │     │  data/logs/pipeline_*.log   │     │
│                └───────────────────────────┘     └─────────────────────────────┘     │
│                                                                                     │
│  ═══════════════════════════════════════════════════════════════════════════════     │
│  FINAL OUTPUT  (ready for ST-GCN training)                                          │
│                                                                                     │
│  processed/                                                                         │
│  ├── poses/               .npy files from ALL datasets combined                     │
│  ├── pose_mapping.csv     video filename → .npy path                                │
│  ├── combined_metadata.csv  full catalog (user, file, gloss, dataset)               │
│  ├── train.csv            80% split  (user, file, gloss, dataset)                   │
│  ├── val.csv              10% split                                                 │
│  ├── test.csv             10% split                                                 │
│  ├── label_map.json       {"hello": 0, "goodbye": 1, ...}                          │
│  └── dataset_stats.json   per-split, per-class video counts                         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Schemas

Full data schemas are documented in [`data_schema.md`](./data_schema.md).

### Summary of Key Entities

| Entity            | Store                  | Purpose                              |
|-------------------|------------------------|--------------------------------------|
| `sign_glosses`    | Aurora PostgreSQL      | Master vocabulary (2,731 signs)      |
| `signers`         | Aurora PostgreSQL      | Signer metadata from datasets        |
| `sign_videos`     | Aurora PostgreSQL      | Raw video metadata + S3 paths        |
| `extracted_poses` | Aurora PostgreSQL      | Per-video pose keypoint metadata     |
| `label_map`       | Aurora PostgreSQL      | Versioned gloss→index mapping        |
| `pipeline_runs`   | Aurora PostgreSQL      | Pipeline execution audit trail       |
| `translations`    | Firestore              | App prediction logs                  |
| `sessions`        | Firestore              | User session tracking                |
| `model_registry`  | DynamoDB               | Deployed model versions              |
| Raw videos        | S3 (raw zone)          | Binary video files                   |
| Pose files        | S3 (processed zone)    | .npy pose keypoint sequences         |
| Model checkpoints | S3 (models bucket)     | .pt files + configs                  |

---

## 5. Pipeline Stages in Detail

### Stage 1: Data Ingestion

**Purpose:** Download raw datasets and store in the data lake (S3 raw zone).

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | ASL Citizen ZIP, WLASL JSON + videos, MS-ASL JSONs + videos |
| **Output**      | Videos in `s3://eyehearu-data-lake/raw/`             |
| **AWS Services**| S3, Lambda (trigger), Glue Crawler                   |
| **Local Code**  | `data/scripts/download_asl_citizen.py`, `download_wlasl.py`, `download_ms_asl.py` |
| **Trigger**     | Manual (new dataset release) or scheduled monthly    |

**Steps:**
1. Download ASL Citizen ZIP from Microsoft (wget/requests)
2. Extract videos, parse metadata CSV (columns: user, filename, gloss)
3. Download WLASL metadata JSON, identify available videos
4. Download MS-ASL metadata JSONs (`MSASL_train.json`, `MSASL_val.json`, `MSASL_test.json`), download corresponding videos
5. Upload to S3 raw zone with dataset/gloss partitioning
6. **Build combined metadata** — merge ASL Citizen CSV + WLASL JSON + MS-ASL JSONs
   into a single `combined_metadata.csv` (user, filename, gloss, dataset)
7. Run Glue Crawler to update Data Catalog

### Stage 2: Data Cleaning

**Purpose:** Validate raw videos, remove corrupted/unusable files.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Raw videos from S3 raw zone                          |
| **Output**      | Validated videos in `s3://eyehearu-data-lake/cleaned/`|
| **AWS Services**| Lambda (per-file validation), Glue (batch cleaning)  |
| **Local Code**  | `data/scripts/pipeline.py` → `stage_clean()`         |
| **Trigger**     | After ingestion stage completes                      |

**Cleaning Rules:**
- Remove videos < 0.3 seconds duration
- Remove videos that cannot be decoded (codec/corruption check)
- Remove videos with resolution < 64x64
- Analyse class balance across the combined metadata; flag classes with < 5 video samples

### Stage 3: Pose Extraction

**Purpose:** Extract pose keypoints from cleaned videos using MediaPipe Holistic.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Cleaned videos from S3 cleaned zone                  |
| **Output**      | `.npy` pose files in S3 processed zone               |
| **AWS Services**| AWS Batch (GPU compute for MediaPipe)                |
| **Local Code**  | `ST-GCN/pose.py`                                     |
| **Trigger**     | After cleaning stage completes                       |

**Extraction Steps:**
1. **Pose Detection** — MediaPipe Holistic extracts 543 keypoints per frame
   (33 pose + 21 left hand + 21 right hand + 468 face landmarks)
2. **Coordinate Extraction** — Store normalized (x, y) for each keypoint
3. **Save** — Write as `.npy` file with shape `(T, 543, 2)` where T = frame count
4. **Mapping File** — Generate `pose_mapping.csv` linking video filenames to `.npy` paths

*Note: Further processing (keypoint selection, normalization to shoulder frame,
padding/downsampling to 128 frames) happens at training time in the dataset
loader (`ST-GCN/asl_citizen_dataset_pose.py`).*

### Stage 4: Data Splitting

**Purpose:** Split **combined** video metadata into train/val/test and generate label maps.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | `combined_metadata.csv` (user, filename, gloss, dataset) |
| **Output**      | train/val/test CSV splits + label_map + stats (in `processed/`) |
| **AWS Services**| Glue (partition management), S3 (organized storage)  |
| **Local Code**  | `data/scripts/pipeline.py` → `stage_split()`         |
| **Trigger**     | After ingestion/cleaning stages complete             |

**Split Ratios:** 80% train / 10% val / 10% test (stratified by class).
Splits are defined at the **video level** — each row in the split CSVs
references a video, not an individual frame or image. Videos from **all
datasets** (ASL Citizen, WLASL, MS-ASL) are shuffled together so the model
trains on the full combined vocabulary.

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
| **Incremental Ingest**| New WLASL videos added      | Weekly              | Expand vocabulary coverage      |
| **Retraining**        | Data drift detected         | Monthly/on-demand   | Model performance degradation   |
| **Analytics ETL**     | Prediction logs accumulate  | Daily (batch)       | Usage dashboards & reporting    |
| **Real-time Inference**| User taps "Capture"        | Per-request         | Core app functionality          |

### Use Case: Initial Model Training

```
Trigger: First-time setup / new target vocabulary defined
Schedule: One-time, then on-demand

1. Download ASL Citizen dataset (Stage 1: Ingest)
2. Download WLASL dataset (Stage 1: Ingest)
3. Validate videos, remove corrupted files (Stage 2: Clean)
4. Extract pose keypoints via MediaPipe Holistic (Stage 3: Pose Extraction)
5. Split video metadata 80/10/10 (Stage 4: Split)
6. Train ST-GCN on pose data on SageMaker (input: 2, 128, 27 tensors)
7. Evaluate and register best model (Stage 5: Report)
```

### Use Case: Restaurant Scenario Enhancement

```
Trigger: Team records additional restaurant-specific signs
Schedule: As needed

1. Record new videos for: eat, drink, hot, cold, more, enough, check
2. Ingest recordings to S3 (Stage 1)
3. Validate new videos (Stage 2)
4. Extract poses for new videos (Stage 3)
5. Merge with existing video metadata
6. Re-split to maintain balanced splits (Stage 4)
7. Fine-tune existing ST-GCN model on expanded data
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
ASL-citizen-code/
└── ST-GCN/                              # Pose-based pipeline
    ├── pose.py                          # MediaPipe Holistic pose extraction
    ├── asl_citizen_dataset_pose.py      # Pose dataset loader (PyTorch)
    ├── pose_transforms.py              # Shear & rotation augmentations
    ├── architecture/
    │   ├── st_gcn.py                   # ST-GCN encoder
    │   ├── graph_utils.py             # Skeleton graph construction
    │   ├── fc.py                       # FC decoder head
    │   └── network.py                 # Encoder + decoder wrapper
    └── README.md

data/
├── scripts/
│   ├── download_asl_citizen.py         # ASL Citizen dataset downloader
│   ├── download_wlasl.py              # WLASL dataset downloader
│   ├── download_ms_asl.py             # MS-ASL dataset downloader
│   └── pipeline.py                     # Main pipeline orchestrator
├── raw/                                # Raw downloaded data (gitignored)
│   ├── asl_citizen/
│   │   ├── videos/
│   │   └── metadata.csv               # user, filename, gloss
│   ├── wlasl/
│   │   ├── videos/
│   │   └── WLASL_v0.3.json            # gloss → instances[]
│   ├── ms_asl/
│   │   ├── videos/
│   │   ├── MSASL_train.json           # [{url, text, signer, label, ...}]
│   │   ├── MSASL_val.json
│   │   └── MSASL_test.json
├── processed/                          # Derived representations (gitignored)
│   ├── combined_metadata.csv           # Unified: user, file, gloss, dataset
│   ├── poses/                          # MediaPipe pose .npy files
│   │   ├── asl_citizen_signer01_hello.npy
│   │   ├── wlasl_12345.npy
│   │   ├── ms_asl_00042.npy
│   │   └── ...
│   ├── pose_mapping.csv               # video filename → .npy path
│   ├── train.csv                       # 80% split (all datasets combined)
│   ├── val.csv                         # 10% split
│   ├── test.csv                        # 10% split
│   ├── label_map.json                  # gloss → integer index
│   └── dataset_stats.json              # per-split, per-class counts
├── metadata/                           # Pipeline metadata + provenance
│   ├── raw_data_catalog.json
│   ├── validated_video_catalog.json
│   ├── dataset_manifest.json
│   └── pipeline_run_*.json
└── logs/
    └── pipeline_*.log
```

### Running the Pipeline

```bash
# Full pipeline (all stages)
cd data/scripts
python pipeline.py --stage all

# Individual stages
python pipeline.py --stage ingest       # Download datasets
python pipeline.py --stage clean        # Validate videos
python pipeline.py --stage pose         # Extract MediaPipe poses → .npy
python pipeline.py --stage split        # Split & generate metadata
python pipeline.py --stage report       # Generate reports

# Pose extraction (standalone)
cd ASL-citizen-code/ST-GCN
python pose.py
```

### Key Code Files

| File                                     | Purpose                                 | Lines |
|------------------------------------------|-----------------------------------------|-------|
| `pipeline.py`                            | Main orchestrator with 5 stages         | ~500  |
| `ST-GCN/pose.py`                         | MediaPipe pose extraction → .npy        | ~70   |
| `ST-GCN/asl_citizen_dataset_pose.py`     | Pose dataset loader + normalization     | ~134  |
| `ST-GCN/pose_transforms.py`             | Shear & rotation augmentations          | ~102  |
| `ST-GCN/architecture/st_gcn.py`          | ST-GCN encoder (graph + temporal conv)  | ~225  |
| `ST-GCN/architecture/graph_utils.py`     | Skeleton graph adjacency construction   | ~154  |
| `ST-GCN/architecture/fc.py`             | Fully-connected decoder head            | ~39   |
| `ST-GCN/architecture/network.py`        | Encoder + decoder wrapper               | ~13   |

---

## 8. Technology Stack

### AWS Services (Production)

| Layer                | Service                  | Purpose                           |
|----------------------|--------------------------|-----------------------------------|
| **Compute**          | AWS Lambda               | Video validation, lightweight tasks|
|                      | AWS Batch                | Pose extraction (MediaPipe GPU)   |
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
| OpenCV        | 4.x     | Video decoding for pose extraction          |
| MediaPipe     | 0.10+   | Holistic pose extraction (543 keypoints)    |
| PyTorch       | 2.2+    | ST-GCN model training and inference         |
| NumPy         | 1.26+   | Pose arrays (.npy), numerical operations    |
| Pandas        | 2.x     | Metadata CSVs, provenance tracking          |
| FastAPI       | 0.110+  | Backend REST API                            |
| scikit-learn  | 1.4+    | Data splitting, evaluation metrics          |
| tqdm          | 4.x     | Progress bars for pipeline stages           |

---

## 9. Next Steps & Future Enhancements

### Implemented (v1.0)
- [x] ASL Citizen dataset download and metadata parsing
- [x] WLASL dataset metadata parsing and filtering
- [x] Video validation (duration, resolution, corruption checks)
- [x] MediaPipe Holistic pose extraction → `.npy` files (543 keypoints per frame)
- [x] ST-GCN model: keypoint selection (27 pts), shoulder-frame normalization, graph convolutions
- [x] Pose augmentations (shear, rotation transforms)
- [x] Train/val/test splitting at video level (80/10/10)
- [x] Label map generation (2,731 ASL Citizen glosses)
- [x] Pipeline orchestration with stage-by-stage execution
- [x] Data quality reporting and provenance tracking
- [x] Comprehensive data schemas (SQL, Document, Object Store)

### Not Yet Implemented (Future Enhancements)

#### Short-Term (Next Sprint)
- [ ] **AWS S3 Integration** — Upload raw/processed data to S3 buckets instead of local storage
- [ ] **AWS Step Functions** — Replace local orchestrator with cloud-native workflow
- [ ] **Additional Pose Augmentations** — Scaling, time-warping, keypoint dropout
- [ ] **Automated Retraining Trigger** — CloudWatch alarm on accuracy drop → auto-retrain

#### Medium-Term (Next Month)
- [ ] **Kinesis Streaming** — Stream real-time predictions to analytics warehouse
- [ ] **Redshift Analytics** — Build prediction analytics warehouse for dashboards
- [ ] **QuickSight Dashboard** — Visual monitoring of model performance and usage
- [ ] **Signer-Stratified Splitting** — Ensure no signer appears in both train and test sets to prevent identity leakage

#### Long-Term (Future Semester)
- [ ] **I3D / SlowFast Video Models** — Evaluate video-based 3D CNN models as an alternative or complement to pose-based ST-GCN
- [ ] **Continuous Learning Pipeline** — Incorporate user feedback (correct/wrong) to improve model over time
- [ ] **Federated Data Collection** — Allow users to opt-in contributing sign videos for underrepresented classes
- [ ] **Multi-Dataset Fusion** — Gloss normalization and conflict resolution across ASL Citizen, WLASL, and MS-ASL (handle synonyms, overlapping labels)
- [ ] **Data Versioning** — Integrate DVC (Data Version Control) for reproducible dataset versions
- [ ] **Cost Optimization** — S3 Intelligent-Tiering, Spot Instances for SageMaker training
- [ ] **Data Privacy Compliance** — Add PII detection and anonymization for any user-contributed data
