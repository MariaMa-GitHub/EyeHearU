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
│ num_frames   int    │                    │
│ split        enum   │  ← train/val/test  │
│ is_valid   boolean  │                    │
│ ingested_at timestamp│       ┌───────────▼─────────────┐
└─────────┬───────────┘        │   LabelMap              │
          │                    │─────────────────────────│
          │                    │ gloss_id   PK FK        │
          │                    │ label_index  int        │
          │                    │ version      varchar    │
          ▼                    │ created_at   timestamp  │
┌─────────────────────┐        └─────────────────────────┘
│   ExtractedPose     │
│─────────────────────│        ┌─────────────────────────┐
│ pose_id    PK       │        │   Signer                │
│ video_id   FK       │        │─────────────────────────│
│ gloss_id   FK       │        │ signer_id    PK         │
│ num_frames   int    │        │ dataset_source          │
│ num_keypoints int   │        │ is_native_signer bool   │
│ file_path    varchar│        │ consent_status          │
│ file_size_bytes int │        └─────────────────────────┘
│ extracted_at timestamp│
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
| `num_train_videos`  | int           |               | Count of training videos for this gloss            |
| `num_val_videos`    | int           |               | Count of validation videos                         |
| `num_test_videos`   | int           |               | Count of test videos                               |
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
| `signer_id`         | varchar(50)   | FK → signers  | Who performed the sign (user column in CSV)        |
| `dataset_source`    | varchar(50)   |               | "asl_citizen", "wlasl", "custom"                   |
| `s3_path`           | varchar(500)  |               | S3 URI: s3://eyehearu-data-lake/raw/...           |
| `duration_sec`      | decimal(6,2)  |               | Video duration in seconds                          |
| `fps`               | decimal(5,1)  |               | Frames per second                                  |
| `num_frames`        | int           |               | Total frame count in the video                     |
| `resolution`        | varchar(20)   |               | e.g., "1920x1080", "640x480"                      |
| `file_size_bytes`   | bigint        |               | Raw file size                                      |
| `split`             | varchar(10)   |               | "train", "val", or "test"                         |
| `is_valid`          | boolean       |               | Passed validation checks                           |
| `ingested_at`       | timestamp     |               | When ingested into the pipeline                    |

### `extracted_poses` — Pose keypoint file metadata

| Column              | Type          | Constraints   | Description                                        |
|---------------------|---------------|---------------|----------------------------------------------------|
| `pose_id`           | varchar(100)  | PK            | Unique pose file identifier                        |
| `video_id`          | varchar(100)  | FK → sign_videos | Source video                                   |
| `gloss_id`          | varchar(50)   | FK → sign_glosses | Inherited from video                           |
| `num_frames`        | int           |               | Temporal length T of the pose sequence             |
| `num_keypoints`     | int           |               | Keypoints per frame (543 for MediaPipe Holistic)   |
| `s3_path`           | varchar(500)  |               | S3 URI for the .npy pose file                     |
| `file_size_bytes`   | bigint        |               | Size of the .npy file                             |
| `extracted_at`      | timestamp     |               | When pose extraction was completed                 |

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
| `clip_hash`      | string    | Hash of the input video clip (for dedup)       |
| `clip_duration_sec` | number | Duration of the submitted video clip           |
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

All binary data (videos, pose files, model checkpoints) is stored in S3.

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
│   │       ├── asl_citizen_metadata.csv    (user, filename, gloss)
│   │       ├── train.csv                   (split subset)
│   │       ├── val.csv
│   │       └── test.csv
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
│   │   └── videos/                         (validated videos only)
│   └── wlasl/
│       └── videos/
│
├── processed/
│   ├── poses/                              (MediaPipe Holistic output)
│   │   ├── signer01_hello.npy              (T, 543, 2) float64
│   │   ├── signer01_goodbye.npy
│   │   └── ...
│   ├── pose_mapping.csv                    (video filename → .npy path)
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
│   └── st_gcn/
│       └── v1.0.0/
│           ├── best_model.pt
│           ├── label_map.json
│           └── config.json
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
| `processed/poses/`        | S3 Standard         | Keep (active training) |
| `metadata/`               | S3 Standard         | Keep indefinitely      |
| `models/checkpoints/`     | S3 Standard         | Keep active versions   |
| `analytics/`              | S3 Standard-IA      | → Glacier after 365 days|

---

## 4. ML Data Schema

### Training Data Layout (Local)

```text
data/
├── raw/
│   └── asl_citizen/
│       ├── videos/                      # Raw video files
│       │   ├── signer01_hello.mp4
│       │   └── ...
│       └── metadata/
│           ├── asl_citizen_metadata.csv  # Full dataset (user, filename, gloss)
│           ├── train.csv                 # Training split
│           ├── val.csv                   # Validation split
│           └── test.csv                  # Test split
│
├── processed/
│   ├── poses/                           # MediaPipe Holistic output
│   │   ├── signer01_hello.npy           # shape (T, 543, 2), float64
│   │   └── ...
│   ├── pose_mapping.csv                 # video filename → .npy path
│   ├── label_map.json                   # gloss → integer index
│   └── dataset_stats.json               # per-split, per-class counts
```

### Metadata CSV Format

All split CSVs share the same schema (one row per video):

| Column     | Type   | Description                          |
|------------|--------|--------------------------------------|
| `user`     | string | Signer / user identifier             |
| `filename` | string | Relative path to video file          |
| `gloss`    | string | Sign label (e.g., "hello", "water")  |

### label_map.json

The label map is auto-generated from the dataset glosses. With the full ASL
Citizen vocabulary this maps 2,731 glosses to integer indices:

```json
{
  "hello": 0,
  "goodbye": 1,
  "water": 2,
  "...": "..."
}
```

### Pose File Format (.npy)

Each `.npy` file stores one video's pose sequence:

| Property       | Value                                       |
|----------------|---------------------------------------------|
| **Shape**      | `(T, 543, 2)` — T varies per video         |
| **Dtype**      | `float64`                                   |
| **Axis 0**     | Time (frames)                               |
| **Axis 1**     | 543 keypoints: 33 pose + 21 R hand + 21 L hand + 468 face |
| **Axis 2**     | (x, y) normalized coordinates (0–1 range)   |

At training time the ST-GCN dataset loader further processes these:
- Selects 27 keypoints (subset of pose + hand landmarks)
- Normalizes to shoulder-centered coordinate frame
- Pads or downsamples to 128 frames
- Transposes to `(2, 128, 27)` — (channels, time, nodes)

### Model Input Tensor Shape

| Input Shape              | Description                          |
|--------------------------|--------------------------------------|
| `(batch, 2, 128, 27)`   | (channels=xy, time, nodes)           |

- **Axis 0** (batch): Mini-batch dimension
- **Axis 1** (2): x and y coordinates
- **Axis 2** (128): Temporal frames (padded or downsampled)
- **Axis 3** (27): Selected keypoints (pose + hand landmarks)

### Label Tensor Format

One-hot encoded vector of shape `(num_classes,)` per sample.

### Model Checkpoint Schema

```text
models/
└── st_gcn/
    ├── best_model.pt           # Best validation accuracy weights
    ├── label_map.json          # Copy of label map (for inference)
    └── config.json             # {num_classes, num_keypoints, graph_strategy, ...}
```

---

## 5. API Request/Response Schema

### POST /api/v1/predict

**Request:** `multipart/form-data`
- `file`: Video clip (MP4/WebM, max 10MB, typically 1–5 seconds)

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
  "latency_ms": 120,
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
- **Mobile camera quality**: No dataset specifically captures from phone cameras at typical framerates
- **Video length variance**: Signing speed varies widely; the pipeline handles this with padding/downsampling to 128 frames in the ST-GCN dataset loader
- **Mitigation**: Supplement with custom-recorded video data from team members
