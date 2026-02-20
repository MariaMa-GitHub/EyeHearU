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
pipeline transforms raw ASL video datasets into preprocessed, fixed-size video
clips suitable for training an **I3D (Inflated 3D ConvNet)** classifier. Videos
are processed as temporal sequences — not cut into individual frames or images.
The I3D model operates directly on RGB video

### Data Sources

| Source       | Type     | Size    | Description                                  |
|-------------|----------|---------|----------------------------------------------|
| ASL Citizen | Videos   | ~84K    | Crowdsourced, 2,731 signs, 52 signers (Microsoft Research) |
| WLASL       | Videos   | ~21K    | YouTube-sourced, 2,000 signs (academic)      |
| MS-ASL      | Videos   | ~25K    | YouTube-sourced, 1,000 signs, 222 signers (Microsoft Research) |
| User App    | Video clips | Growing | Real-time predictions from the mobile app |

### Pipeline Summary

```
 ASL Citizen ─┐                                                      ┌─▶ train.csv (80%)
 WLASL ───────┼──▶ Combine ──▶ Validate ──▶ Preprocess clips ──▶ Split ──▶ val.csv   (10%)
 MS-ASL ──────┘    metadata     videos       → 64-frame .mp4s          └─▶ test.csv  (10%)
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
                       ▼            │  processed_clips     │    ┌────────┴────────┐
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
and combined into **one unified dataset** of preprocessed video clips for
I3D training.

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
│  STAGE 3 ─ VIDEO PREPROCESSING                                                      │
│                ┌───────────────────────────┐                                        │
│                │  CLIP PREPROCESSING        │                                        │
│                │                           │                                        │
│                │  For each video in        │                                        │
│                │  combined_metadata.csv:   │     ┌─────────────────────────────┐     │
│                │                           │     │  processed/clips/           │     │
│                │  • Open video with OpenCV │────▶│    {split}/{gloss}/*.mp4    │     │
│                │  • Center-crop temporally │     │                             │     │
│                │    to max 64 frames       │     │  Each clip:                 │     │
│                │  • Skip every Nth frame   │     │    64 frames, 256×256 RGB   │     │
│                │    for long videos        │     │    ~100–500 KB per clip     │     │
│                │  • Resize frames to       │     │                             │     │
│                │    256×256 pixels         │     │  processed/                 │     │
│                │  • Write normalised .mp4  │────▶│    processed_clips.csv      │     │
│                │                           │     │  (clip_id, gloss, split,    │     │
│                │  Skip if clip exists      │     │   clip_path, num_frames)    │     │
│                │  (incremental)            │     │                             │     │
│                └───────────────────────────┘     └─────────────────────────────┘     │
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
│                │  • Clip file count + size │     │    pipeline_run_*.json      │     │
│                │  • Per-split video counts │     │                             │     │
│                │  • Class counts per split │     │  Logged to console +        │     │
│                │  • Error summary          │     │  data/logs/pipeline_*.log   │     │
│                └───────────────────────────┘     └─────────────────────────────┘     │
│                                                                                     │
│  ═══════════════════════════════════════════════════════════════════════════════     │
│  FINAL OUTPUT  (ready for I3D training)                                              │
│                                                                                     │
│  processed/                                                                         │
│  ├── clips/               preprocessed 64-frame, 256×256 .mp4 clips                │
│  │   ├── train/{gloss}/*.mp4                                                       │
│  │   ├── val/{gloss}/*.mp4                                                         │
│  │   └── test/{gloss}/*.mp4                                                        │
│  ├── processed_clips.csv  clip_id, gloss, split, clip_path, num_frames              │
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
| `sign_glosses`      | Aurora PostgreSQL      | Master vocabulary (2,731 signs)      |
| `signers`           | Aurora PostgreSQL      | Signer metadata from datasets        |
| `sign_videos`       | Aurora PostgreSQL      | Raw video metadata + S3 paths        |
| `processed_clips`   | Aurora PostgreSQL      | Per-video preprocessed clip metadata |
| `label_map`         | Aurora PostgreSQL      | Versioned gloss→index mapping        |
| `pipeline_runs`     | Aurora PostgreSQL      | Pipeline execution audit trail       |
| `translations`      | Firestore              | App prediction logs                  |
| `sessions`          | Firestore              | User session tracking                |
| `model_registry`    | DynamoDB               | Deployed model versions              |
| Raw videos          | S3 (raw zone)          | Binary video files                   |
| Preprocessed clips  | S3 (processed zone)    | 64-frame, 256×256 .mp4 clips        |
| Model checkpoints   | S3 (models bucket)     | .pt files + configs                  |

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

### Stage 3: Video Preprocessing

**Purpose:** Convert cleaned, variable-length raw videos into fixed-size clips
(64 frames, 256×256 px) ready for I3D training.

| Aspect          | Detail                                               |
|-----------------|------------------------------------------------------|
| **Input**       | Cleaned videos from S3 cleaned zone                  |
| **Output**      | `.mp4` clips in `s3://eyehearu-data-lake/processed/clips/` |
| **AWS Services**| AWS Batch (CPU compute — no GPU needed)              |
| **Local Code**  | `data/scripts/preprocess_clips.py`                   |
| **Trigger**     | After cleaning stage completes                       |

**Preprocessing Steps:**
1. **Read video** — Open with OpenCV, get total frame count and FPS
2. **Temporal sampling** — Center-crop temporally and uniformly sample up to
   64 frames. For long videos (≥96 frames), skip every 2nd or 3rd frame so the
   64-frame window covers the full sign duration
3. **Spatial resize** — Resize each frame to 256×256 pixels (upscale small
   frames, downscale large frames)
4. **Normalize pixels** — Scale to `[-1, 1]` range: `(pixel / 255) * 2 - 1`
5. **Write clip** — Save as `.mp4` under `processed/clips/{split}/{gloss}/`
6. **Record metadata** — Append to `processed_clips.csv` (clip_id, gloss,
   signer_id, split, source, num_frames, height, width, clip_path)

*Note: Pixel normalization is applied at training time by the dataset loader.
The stored `.mp4` clips contain standard 8-bit RGB frames. Padding (if the
video has fewer than 64 frames) is also handled at training time.*

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
4. Preprocess clips: sample 64 frames, resize 256×256 (Stage 3: Preprocess)
5. Split video metadata 80/10/10 (Stage 4: Split)
6. Train I3D on video clips on SageMaker (input: 3, 64, 256, 256 tensors)
7. Evaluate and register best model (Stage 5: Report)
```

### Use Case: Restaurant Scenario Enhancement

```
Trigger: Team records additional restaurant-specific signs
Schedule: As needed

1. Record new videos for: eat, drink, hot, cold, more, enough, check
2. Ingest recordings to S3 (Stage 1)
3. Validate new videos (Stage 2)
4. Preprocess new videos into 64-frame clips (Stage 3)
5. Merge with existing video metadata
6. Re-split to maintain balanced splits (Stage 4)
7. Fine-tune existing I3D model on expanded data
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
└── I3D/                                 # Video-based pipeline (I3D)
    ├── pytorch_i3d.py                   # I3D architecture (pretrained on Kinetics-400)
    ├── aslcitizen_dataset.py            # Video dataset loader (PyTorch)
    ├── videotransforms.py               # Random crop, center crop augmentations
    ├── aslcitizen_training.py           # Training script
    ├── aslcitizen_testing.py            # Evaluation + confusion matrix
    └── README.md

data/
├── scripts/
│   ├── download_asl_citizen.py         # ASL Citizen dataset downloader
│   ├── download_wlasl.py              # WLASL dataset downloader
│   ├── download_ms_asl.py             # MS-ASL dataset downloader
│   ├── preprocess_clips.py            # Trim → sample 64 frames → resize 256×256 → .mp4
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
│   ├── clips/                          # Preprocessed video clips (64 frames, 256×256)
│   │   ├── train/
│   │   │   ├── hello/*.mp4
│   │   │   ├── goodbye/*.mp4
│   │   │   └── ...
│   │   ├── val/{gloss}/*.mp4
│   │   └── test/{gloss}/*.mp4
│   ├── processed_clips.csv             # clip_id, gloss, split, clip_path, num_frames
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
python pipeline.py --stage preprocess   # Trim, sample 64 frames, resize 256×256 → .mp4
python pipeline.py --stage split        # Split & generate metadata
python pipeline.py --stage report       # Generate reports

# Clip preprocessing (standalone)
cd data/scripts
python preprocess_clips.py
```

### Key Code Files

| File                                     | Purpose                                 | Lines |
|------------------------------------------|-----------------------------------------|-------|
| `pipeline.py`                            | Main orchestrator with 5 stages         | ~500  |
| `preprocess_clips.py`                    | Temporal sampling + resize → .mp4 clips | ~100  |
| `I3D/aslcitizen_dataset.py`             | Video dataset loader + padding          | ~177  |
| `I3D/videotransforms.py`                | Random crop, center crop augmentations  | ~50   |
| `I3D/pytorch_i3d.py`                    | I3D architecture (Kinetics-400 pretrained) | ~400 |
| `I3D/aslcitizen_training.py`            | Training loop + validation              | ~200  |
| `I3D/aslcitizen_testing.py`             | Evaluation + confusion matrix analysis  | ~200  |

---

## 8. Technology Stack

### AWS Services (Production)

| Layer                | Service                  | Purpose                           |
|----------------------|--------------------------|-----------------------------------|
| **Compute**          | AWS Lambda               | Video validation, lightweight tasks|
|                      | AWS Batch                | Video preprocessing (CPU)         |
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
| OpenCV        | 4.x     | Video decoding, frame sampling, resizing    |
| PyTorch       | 2.2+    | I3D model training and inference            |
| NumPy         | 1.26+   | Frame array manipulation                    |
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
- [x] Video preprocessing: temporal sampling (64 frames), spatial resize (256×256), write .mp4 clips
- [x] I3D model: Inflated 3D ConvNet pretrained on Kinetics-400, operates directly on RGB video
- [x] Video augmentations (random crop, center crop, frame padding)
- [x] Train/val/test splitting at video level (80/10/10)
- [x] Label map generation (2,731 ASL Citizen glosses)
- [x] Pipeline orchestration with stage-by-stage execution
- [x] Data quality reporting and provenance tracking
- [x] Comprehensive data schemas (SQL, Document, Object Store)

### Not Yet Implemented (Future Enhancements)

#### Short-Term (Next Sprint)
- [ ] **AWS S3 Integration** — Upload raw/processed data to S3 buckets instead of local storage
- [ ] **AWS Step Functions** — Replace local orchestrator with cloud-native workflow
- [ ] **Additional Video Augmentations** — Color jitter, time-warping, random erasing
- [ ] **Automated Retraining Trigger** — CloudWatch alarm on accuracy drop → auto-retrain

#### Medium-Term (Next Month)
- [ ] **Kinesis Streaming** — Stream real-time predictions to analytics warehouse
- [ ] **Redshift Analytics** — Build prediction analytics warehouse for dashboards
- [ ] **QuickSight Dashboard** — Visual monitoring of model performance and usage
- [ ] **Signer-Stratified Splitting** — Ensure no signer appears in both train and test sets to prevent identity leakage

#### Long-Term (Future Semester)
- [ ] **SlowFast / Video Swin Transformer** — Evaluate newer video architectures as complements to I3D
- [ ] **Continuous Learning Pipeline** — Incorporate user feedback (correct/wrong) to improve model over time
- [ ] **Federated Data Collection** — Allow users to opt-in contributing sign videos for underrepresented classes
- [ ] **Multi-Dataset Fusion** — Gloss normalization and conflict resolution across ASL Citizen, WLASL, and MS-ASL (handle synonyms, overlapping labels)
- [ ] **Data Versioning** — Integrate DVC (Data Version Control) for reproducible dataset versions
- [ ] **Cost Optimization** — S3 Intelligent-Tiering, Spot Instances for SageMaker training
- [ ] **Data Privacy Compliance** — Add PII detection and anonymization for any user-contributed data
