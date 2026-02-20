# Data Schema — Eye Hear U

This document describes all data models, storage schemas, and relationships used
across the Eye Hear U system. Modeled after the restaurant-review example schema
but tailored for a real-time ASL-to-English translation system.

---

## Entity Relationship Overview

```text
┌─────────────────────┐        ┌─────────────────────────┐
│   SignVideo          │        │   SignGloss              │
│─────────────────────│        │─────────────────────────│
│ video_id   PK       │        │ gloss_id   PK           │
│ gloss_id   FK  ─────┼───────▶│ gloss_name              │
│ dataset_source      │        │ category                │
│ signer_id  FK       │        │ difficulty              │
│ duration_sec        │        │ reference_video_url     │
│ resolution          │        │ num_samples             │
│ fps                 │        │ created      timestamp  │
│ file_path           │        └─────────────────────────┘
│ file_size_bytes     │                    │
│ is_valid   boolean  │                    │
│ ingested_at timestamp│       ┌───────────▼─────────────┐
└─────────┬───────────┘        │   LabelMap              │
          │                    │─────────────────────────│
          │                    │ gloss_id   PK FK        │
          ▼                    │ label_index  int        │
┌─────────────────────┐        │ version      varchar    │
│   ExtractedFrame    │        │ created_at   timestamp  │
│─────────────────────│        └─────────────────────────┘
│ frame_id   PK       │
│ video_id   FK       │
│ gloss_id   FK       │        ┌─────────────────────────┐
│ frame_index  int    │        │   Signer                │
│ file_path           │        │─────────────────────────│
│ width      int      │        │ signer_id    PK         │
│ height     int      │        │ dataset_source          │
│ is_blurry  boolean  │        │ is_native_signer bool   │
│ is_black   boolean  │        │ consent_status          │
│ hand_detected bool  │        └─────────────────────────┘
│ hash       varchar  │
│ extracted_at timestamp│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   ProcessedImage    │
│─────────────────────│
│ image_id   PK       │
│ frame_id   FK       │
│ gloss_id   FK       │
│ split      enum     │  ← train / val / test
│ file_path           │
│ width      int      │  ← always 224
│ height     int      │  ← always 224
│ hand_cropped bool   │
│ normalized  bool    │
│ processed_at timestamp│
└─────────────────────┘
```

---

## 1. SQL Data Store (Amazon Aurora PostgreSQL)

These tables store structured metadata about the ML training data pipeline. In
production, hosted on **Amazon Aurora PostgreSQL** (Serverless v2). Locally
represented as JSON/CSV files in `data/metadata/`.

### `sign_glosses` — Master vocabulary table

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `gloss_id`          | varchar(50)   | PK            | Unique identifier (e.g., "hello", "A", "1")        |
| `gloss_name`        | varchar(100)  |               | Human-readable display name                        |
| `category`          | varchar(50)   |               | Grouping: greeting, medical, restaurant, letter, number |
| `difficulty`        | int           |               | 1=easy, 2=medium, 3=hard                          |
| `reference_video_url`| varchar(500) | NULL          | URL to a reference video for this sign             |
| `num_train_samples` | int           |               | Count of training images for this gloss            |
| `num_val_samples`   | int           |               | Count of validation images                         |
| `num_test_samples`  | int           |               | Count of test images                               |
| `created`           | timestamp     |               | When this gloss was added to the system            |
| `updated`           | timestamp     |               | Last metadata update                               |

### `signers` — Signer demographic information

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `signer_id`         | varchar(50)   | PK            | Unique signer identifier                           |
| `dataset_source`    | varchar(50)   |               | Source dataset: "asl_citizen", "wlasl", "custom"   |
| `is_native_signer`  | boolean       |               | Whether the signer is a native ASL user            |
| `consent_status`    | varchar(20)   |               | IRB consent: "approved", "pending"                 |
| `added_at`          | timestamp     |               | When this signer was added                         |

### `sign_videos` — Raw video metadata

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `video_id`          | varchar(100)  | PK            | Unique video identifier                            |
| `gloss_id`          | varchar(50)   | FK → sign_glosses | Which sign this video demonstrates            |
| `signer_id`         | varchar(50)   | FK → signers  | Who performed the sign                             |
| `dataset_source`    | varchar(50)   |               | "asl_citizen", "wlasl", "custom"                   |
| `s3_path`           | varchar(500)  |               | S3 URI: s3://eyehearu-data-lake/raw/...           |
| `duration_sec`      | decimal(6,2)  |               | Video duration in seconds                          |
| `fps`               | decimal(5,1)  |               | Frames per second                                  |
| `resolution`        | varchar(20)   |               | e.g., "1920x1080", "640x480"                      |
| `file_size_bytes`   | bigint        |               | Raw file size                                      |
| `is_valid`          | boolean       |               | Passed validation checks                           |
| `ingested_at`       | timestamp     |               | When ingested into the pipeline                    |

### `extracted_frames` — Frame-level metadata

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `frame_id`          | varchar(100)  | PK            | Unique frame identifier                            |
| `video_id`          | varchar(100)  | FK → sign_videos | Source video                                   |
| `gloss_id`          | varchar(50)   | FK → sign_glosses | Inherited from video                           |
| `frame_index`       | int           |               | Position in the video (0-indexed)                  |
| `s3_path`           | varchar(500)  |               | S3 URI for the frame image                        |
| `width`             | int           |               | Original frame width (pixels)                      |
| `height`            | int           |               | Original frame height (pixels)                     |
| `is_blurry`         | boolean       |               | Laplacian variance < threshold                     |
| `is_black`          | boolean       |               | Mean pixel intensity < 10                          |
| `hand_detected`     | boolean       |               | MediaPipe detected a hand                          |
| `perceptual_hash`   | varchar(20)   |               | For deduplication                                  |
| `extracted_at`      | timestamp     |               | When the frame was extracted                       |

### `processed_images` — Final training-ready images

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `image_id`          | varchar(100)  | PK            | Unique processed image identifier                  |
| `frame_id`          | varchar(100)  | FK → extracted_frames | Source frame                               |
| `gloss_id`          | varchar(50)   | FK → sign_glosses | Sign label                                     |
| `split`             | varchar(10)   |               | "train", "val", or "test"                         |
| `s3_path`           | varchar(500)  |               | S3 URI for processed image                        |
| `width`             | int           |               | Always 224 (target size)                          |
| `height`            | int           |               | Always 224 (target size)                          |
| `hand_cropped`      | boolean       |               | Whether hand region was cropped                    |
| `normalized`        | boolean       |               | Whether pixel values were normalized               |
| `processed_at`      | timestamp     |               | When processing was applied                        |

### `label_map` — Gloss-to-index mapping (versioned)

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `gloss_id`          | varchar(50)   | PK, FK        | Sign label                                         |
| `label_index`       | int           | UNIQUE        | Integer class index for the model                  |
| `version`           | varchar(20)   |               | Dataset version (e.g., "20260214")                 |
| `created_at`        | timestamp     |               | When this mapping was created                      |

### `pipeline_runs` — Pipeline execution metadata

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `run_id`            | varchar(50)   | PK            | Unique pipeline run identifier                     |
| `start_time`        | timestamp     |               | Pipeline start time                                |
| `end_time`          | timestamp     | NULL          | Pipeline end time                                  |
| `duration_sec`      | decimal(10,2) |               | Total duration                                     |
| `stages_completed`  | text          |               | JSON array of completed stages                     |
| `status`            | varchar(20)   |               | "running", "completed", "failed"                   |
| `trigger`           | varchar(50)   |               | "manual", "scheduled", "new_data"                  |
| `error_message`     | text          | NULL          | Error details if failed                            |

---

## 2. Document Store (Amazon DynamoDB / Firebase Firestore)

These collections store semi-structured operational data. In production,
**Amazon DynamoDB** for pipeline metadata and **Firebase Firestore** for
app-facing data.

### `translations` collection (Firestore)

Stores each prediction made by the app (for analytics and history).

| Field            | Type      | Description                                    |
|------------------|-----------|------------------------------------------------|
| `session_id`     | string    | Anonymous session identifier                   |
| `predicted_sign` | string    | The top-1 predicted ASL sign label             |
| `confidence`     | number    | Model confidence score (0.0 – 1.0)            |
| `top_k`          | array     | Top-k predictions [{sign, confidence}]         |
| `timestamp`      | timestamp | When the prediction was made                   |
| `image_hash`     | string    | Hash of the input image (for dedup, no raw img)|
| `feedback`       | string    | Optional user feedback ("correct" / "wrong")   |
| `model_version`  | string    | Which model version produced this prediction   |
| `latency_ms`     | number    | Backend inference latency in milliseconds      |

### `sessions` collection (Firestore)

| Field          | Type      | Description                          |
|----------------|-----------|--------------------------------------|
| `session_id`   | string    | Unique session identifier            |
| `device_info`  | map       | {model, os_version, screen_size}     |
| `created_at`   | timestamp | Session start time                   |
| `last_active`  | timestamp | Last activity time                   |
| `total_predictions` | number | Total predictions in this session  |

### `model_registry` collection (DynamoDB)

Tracks deployed model versions and their performance.

| Field              | Type      | Description                                |
|--------------------|-----------|--------------------------------------------|
| `model_version`    | string    | PK — Version identifier (e.g., "v1.0.0")  |
| `checkpoint_s3`    | string    | S3 path to model checkpoint (.pt file)     |
| `label_map_s3`     | string    | S3 path to label_map.json                  |
| `dataset_version`  | string    | Which dataset version was used for training|
| `accuracy`         | number    | Validation accuracy                        |
| `top5_accuracy`    | number    | Top-5 validation accuracy                  |
| `num_classes`      | number    | Number of sign classes                     |
| `deployed_at`      | timestamp | When deployed to production                |
| `status`           | string    | "active", "staging", "retired"             |

### `vocabulary` collection (Firestore — future learning mode)

| Field        | Type      | Description                               |
|--------------|-----------|-------------------------------------------|
| `gloss`      | string    | The sign label (e.g., "hello")            |
| `category`   | string    | Grouping (greeting, medical, restaurant)  |
| `difficulty` | number    | 1=easy, 2=medium, 3=hard                 |
| `video_url`  | string    | Reference video showing the sign          |

---

## 3. Object Store (Amazon S3)

All binary data (videos, images, model checkpoints) is stored in S3.

### Bucket Structure

```text
s3://eyehearu-data-lake/
├── raw/
│   ├── asl_citizen/
│   │   ├── videos/
│   │   │   ├── signer01_hello.mp4
│   │   │   ├── signer01_goodbye.mp4
│   │   │   └── ...
│   │   └── metadata/
│   │       └── asl_citizen_metadata.csv
│   ├── wlasl/
│   │   ├── videos/
│   │   │   ├── 00001.mp4
│   │   │   └── ...
│   │   └── WLASL_v0.3.json
│   └── custom/
│       └── team_recordings/
│
├── cleaned/
│   ├── asl_citizen/
│   │   └── validated_frames/
│   └── wlasl/
│       └── validated_frames/
│
├── processed/
│   ├── images/
│   │   ├── train/
│   │   │   ├── hello/
│   │   │   │   ├── frame_000001.jpg    (224x224, normalized)
│   │   │   │   └── ...
│   │   │   ├── goodbye/
│   │   │   └── ...
│   │   ├── val/
│   │   │   └── (same structure)
│   │   └── test/
│   │       └── (same structure)
│   ├── label_map.json
│   └── dataset_stats.json
│
└── metadata/
    ├── raw_data_catalog.json
    ├── provenance/
    │   └── asl_citizen_provenance.csv
    ├── quality_reports/
    │   └── asl_citizen_quality_report.json
    └── pipeline_runs/
        └── pipeline_run_20260214_120000.json

s3://eyehearu-models/
├── checkpoints/
│   ├── v1.0.0/
│   │   ├── best_model.pt
│   │   ├── label_map.json
│   │   └── config.json
│   └── v1.1.0/
│       └── ...
└── exports/
    └── (optimized models for deployment)

s3://eyehearu-analytics/
├── prediction_logs/
│   └── (partitioned by date)
└── dashboards/
    └── (aggregated metrics)
```

### S3 Lifecycle Policies

| Prefix                    | Storage Class       | Transition             |
|---------------------------|---------------------|------------------------|
| `raw/`                    | S3 Standard         | → Glacier after 90 days|
| `cleaned/`                | S3 Standard-IA      | → Glacier after 180 days|
| `processed/`              | S3 Standard         | Keep (active training) |
| `metadata/`               | S3 Standard         | Keep indefinitely      |
| `models/checkpoints/`     | S3 Standard         | Keep active versions   |
| `analytics/`              | S3 Standard-IA      | → Glacier after 365 days|

---

## 4. ML Data Schema

### Training Data Layout (Local)

```text
data/processed/
├── images/
│   ├── train/
│   │   ├── hello/
│   │   │   ├── signer01_frame_000042.jpg
│   │   │   ├── signer02_frame_000018.jpg
│   │   │   └── ...
│   │   ├── goodbye/
│   │   └── ...    (62 class directories)
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
├── label_map.json          # {"A": 0, "B": 1, ..., "hello": 26, ...}
└── dataset_stats.json      # per-split, per-class counts
```

### label_map.json

```json
{
  "1": 0, "10": 1, "2": 2, "3": 3, "4": 4, "5": 5,
  "6": 6, "7": 7, "8": 8, "9": 9,
  "A": 10, "B": 11, "C": 12, "D": 13, "E": 14, "F": 15,
  "G": 16, "H": 17, "I": 18, "J": 19, "K": 20, "L": 21,
  "M": 22, "N": 23, "O": 24, "P": 25, "Q": 26, "R": 27,
  "S": 28, "T": 29, "U": 30, "V": 31, "W": 32, "X": 33,
  "Y": 34, "Z": 35,
  "allergic": 36, "bathroom": 37, "check": 38, "cold": 39,
  "doctor": 40, "drink": 41, "eat": 42, "emergency": 43,
  "enough": 44, "food": 45, "goodbye": 46, "hello": 47,
  "help": 48, "hot": 49, "hurt": 50, "medicine": 51,
  "more": 52, "my": 53, "name": 54, "no": 55, "pain": 56,
  "please": 57, "sick": 58, "sorry": 59, "stop": 60,
  "thank you": 61, "wait": 62, "water": 63, "yes": 64, "you": 65, "your": 66,
  "me": 67
}
```

### Model Checkpoint Schema

```text
ml/checkpoints/
├── best_model.pt           # Best validation accuracy weights
├── label_map.json          # Copy of label map (for inference)
└── config.json             # Training config snapshot:
                            #   {backbone, num_classes, image_size,
                            #    dataset_version, training_epochs, ...}
```

---

## 5. API Request/Response Schema

### POST /api/v1/predict

**Request:** `multipart/form-data`
- `file`: JPEG/PNG image file (max 5MB)

**Response:**
```json
{
  "sign": "hello",
  "confidence": 0.92,
  "top_k": [
    {"sign": "hello", "confidence": 0.92},
    {"sign": "help",  "confidence": 0.05},
    {"sign": "hi",    "confidence": 0.02}
  ],
  "model_version": "v1.0.0",
  "latency_ms": 45,
  "message": null
}
```

### GET /api/v1/history?session_id=abc123

**Response:**
```json
{
  "session_id": "abc123",
  "predictions": [
    {
      "predicted_sign": "water",
      "confidence": 0.88,
      "timestamp": "2026-02-14T15:30:00Z",
      "feedback": "correct"
    }
  ],
  "total_count": 15
}
```

---

## 6. Dataset Comparison & Gap Analysis

### Available Datasets

| Dataset      | Glosses | Videos | Signers | Quality Notes                           |
|--------------|---------|--------|---------|----------------------------------------|
| ASL Citizen  | 2,731   | ~84K   | 52      | Crowdsourced, diverse backgrounds, IRB-approved |
| WLASL        | 2,000   | ~21K   | 100+    | YouTube-sourced, many links broken     |
| MS-ASL       | 1,000   | ~25K   | 222     | Cleaned YouTube data, good diversity   |
| ASL-LEX      | 2,723   | ~2.7K  | 1       | Single signer, very clean              |

### Coverage Analysis for Target Vocabulary

| Category    | Target Signs | Expected Coverage (ASL Citizen) | Gap Mitigation          |
|-------------|-------------|--------------------------------|------------------------|
| Letters A–Z | 26          | ~26 (all common)               | Supplement with WLASL  |
| Numbers 1–10| 10          | ~8–10                          | Custom recordings      |
| Greetings   | 10          | ~8–10                          | WLASL supplement       |
| Restaurant  | 7           | ~5–7                           | Custom recordings      |
| Medical     | 6           | ~4–6                           | Custom recordings      |
| Common      | 3           | ~3                             | Good coverage expected |

### Key Gaps

- **Broken links**: WLASL videos are YouTube-sourced; many are now unavailable
- **Background diversity**: Most datasets have clean backgrounds, unlike real-world phone use
- **Mobile camera quality**: No dataset specifically captures from phone cameras
- **Our target vocab**: Not all 62 target signs may be in available datasets
- **Mitigation**: Supplement with custom-recorded data from team members
