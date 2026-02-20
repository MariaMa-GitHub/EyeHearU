# Data Schemas — Eye Hear U

This document defines the data schemas used at every stage of the data pipeline, from raw ingestion through to the final processed dataset consumed by the video classifier, and the backend prediction store.

---

## Entity Relationship Overview

```
┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│ IngestedASLCitizen  │       │   IngestedWLASL     │       │   IngestedMSASL     │
│─────────────────────│       │─────────────────────│       │─────────────────────│
│ clip_id      PK     │       │ clip_id      PK     │       │ clip_id      PK     │
│ gloss               │       │ gloss               │       │ gloss               │
│ signer_id           │       │ signer_id           │       │ signer_id           │
│ split               │       │ split               │       │ split               │
│ source              │       │ source              │       │ source              │
│ src_path            │       │ frame_start         │       │ start_time          │
│                     │       │ frame_end           │       │ end_time            │
│                     │       │ src_path            │       │ src_path            │
└────────┬────────────┘       └────────┬────────────┘       └────────┬────────────┘
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       │  preprocess_clips.py
                                       ▼
                            ┌─────────────────────┐
                            │   ProcessedClip     │
                            │─────────────────────│
                            │ clip_id      PK     │
                            │ gloss     FK ───────┼──┐
                            │ signer_id           │  │
                            │ split               │  │
                            │ source              │  │
                            │ num_frames          │  │
                            │ height              │  │
                            │ width               │  │
                            │ clip_path           │  │
                            └─────────────────────┘  │
                                                     │
              ┌─────────────────────┐                │
              │     LabelMap        │◄───────────────┘
              │─────────────────────│
              │ gloss        PK     │
              │ label_index         │
              └─────────────────────┘

              ┌─────────────────────┐       ┌─────────────────────┐
              │   Translation       │       │   DatasetStats      │
              │   (Firestore)       │       │─────────────────────│
              │─────────────────────│       │ num_classes          │
              │ doc_id       PK     │       │ total_clips          │
              │ session_id          │       │ splits               │
              │ predicted_sign      │       │ source_breakdown     │
              │ confidence          │       │ per_class            │
              │ top_k               │       └─────────────────────┘
              │ timestamp           │
              │ feedback            │
              └─────────────────────┘
```

> Raw video files are stored in S3 (`s3://{env}-data/raw/`).
> Processed clips are stored in S3 (`s3://{env}-data/processed/clips/`).
> Prediction logs are stored in Firestore (document store).

---

## 1. Ingested Records

After the ingestion scripts run, each dataset source produces a normalised CSV. These are staging tables — one per source — that feed into the preprocessing step.

### 1.1 IngestedASLCitizen — `ingested_asl_citizen.csv`

| Field     | Type     | Constraints | Description                            |
|-----------|----------|-------------|----------------------------------------|
| clip_id   | varchar  | **PK**      | Unique identifier (original filename)  |
| gloss     | varchar  |             | English label for the ASL sign (lower) |
| signer_id | varchar  |             | Anonymous signer identifier            |
| split     | varchar  |             | Official split: `train`, `val`, `test` |
| source    | varchar  |             | Always `"asl_citizen"`                 |
| src_path  | varchar  |             | Absolute path to the raw `.mp4` file   |

### 1.2 IngestedWLASL — `ingested_wlasl.csv`

| Field       | Type     | Constraints | Description                                 |
|-------------|----------|-------------|---------------------------------------------|
| clip_id     | varchar  | **PK**      | `"wlasl_{video_id}"`                        |
| gloss       | varchar  |             | English label for the ASL sign (lower)      |
| signer_id   | varchar  |             | Signer identifier (when available)          |
| split       | varchar  |             | Annotated split: `train`, `val`, `test`     |
| source      | varchar  |             | Always `"wlasl"`                            |
| frame_start | int      |             | Start frame of the sign in the source video |
| frame_end   | int      |             | End frame (`-1` = end of video)             |
| src_path    | varchar  |             | Absolute path to the raw `.mp4` file        |

### 1.3 IngestedMSASL — `ingested_msasl.csv`

| Field      | Type      | Constraints | Description                             |
|------------|-----------|-------------|-----------------------------------------|
| clip_id    | varchar   | **PK**      | `"msasl_{split}_{idx}"`                 |
| gloss      | varchar   |             | English label for the ASL sign (lower)  |
| signer_id  | varchar   |             | Signer identifier                       |
| split      | varchar   |             | Annotated split: `train`, `val`, `test` |
| source     | varchar   |             | Always `"msasl"`                        |
| start_time | float     |             | Start timestamp (seconds) within source |
| end_time   | float     |             | End timestamp (seconds, `-1` = end)     |
| src_path   | varchar   |             | Absolute path to the raw `.mp4` file    |

---

## 2. ProcessedClip — `processed_clips.csv`

After `preprocess_clips.py` trims, resamples, and resizes every video, all sources are unified into a single master table.

| Field      | Type     | Constraints | Description                                      |
|------------|----------|-------------|--------------------------------------------------|
| clip_id    | varchar  | **PK**      | Same as ingested id                              |
| gloss      | varchar  | **FK** → LabelMap.gloss | English label (lower-cased)          |
| signer_id  | varchar  |             | Signer identifier                                |
| split      | varchar  |             | `train`, `val`, or `test`                        |
| source     | varchar  |             | `asl_citizen`, `wlasl`, or `msasl`               |
| num_frames | int      |             | Always `16` (configured in `pipeline_config.py`) |
| height     | int      |             | Always `224`                                     |
| width      | int      |             | Always `224`                                     |
| clip_path  | varchar  |             | Path to the processed `.mp4` on disk / S3        |

### File layout on disk / S3

```
data/processed/                          s3://{env}-data/processed/
├── clips/                               ├── clips/
│   ├── train/                           │   ├── train/
│   │   ├── hello/                       │   │   ├── hello/
│   │   │   ├── wlasl_12345.mp4          │   │   │   └── ...
│   │   │   └── asl_citizen_hello_001.mp4│   │   └── ...
│   │   └── ...                          │   ├── val/
│   ├── val/                             │   │   └── ...
│   │   └── ...                          │   └── test/
│   └── test/                            │       └── ...
│       └── ...                          ├── processed_clips.csv
├── processed_clips.csv                  ├── label_map.json
├── label_map.json                       └── dataset_stats.json
└── dataset_stats.json
```

---

## 3. LabelMap — `label_map.json`

Maps every gloss to a contiguous integer index used by the classifier. Generated by `build_unified_dataset.py`.

| Field       | Type    | Constraints | Description                          |
|-------------|---------|-------------|--------------------------------------|
| gloss       | varchar | **PK**      | English label (lower-cased)          |
| label_index | int     | **UNIQUE**  | Contiguous class index (`0` to `N-1`) |

```json
{
  "about": 0,
  "above": 1,
  "accept": 2,
  "...": "...",
  "zero": 1999
}
```

---

## 4. DatasetStats — `dataset_stats.json`

Aggregate statistics over the entire processed dataset. One record per build.

| Field            | Type   | Constraints | Description                                      |
|------------------|--------|-------------|--------------------------------------------------|
| num_classes      | int    |             | Total number of unique glosses                   |
| total_clips      | int    |             | Total number of processed clips                  |
| splits           | object |             | Clip counts per split (`train`, `val`, `test`)   |
| source_breakdown | object |             | Clip counts per source dataset                   |
| per_class        | object |             | Per-gloss clip counts by split                   |

```json
{
  "num_classes": 2000,
  "total_clips": 95000,
  "splits": {
    "train": 68000,
    "val": 12000,
    "test": 15000
  },
  "source_breakdown": {
    "asl_citizen": 75000,
    "wlasl": 12000,
    "msasl": 8000
  },
  "per_class": {
    "hello": { "train": 85, "val": 12, "test": 20 },
    "...": {}
  }
}
```

---

## 5. Translation — Firestore `translations` collection

The backend stores prediction logs and user feedback in Firestore (document store).

| Field          | Type      | Constraints | Description                                 |
|----------------|-----------|-------------|---------------------------------------------|
| doc_id         | varchar   | **PK**      | Auto-generated Firestore document ID        |
| session_id     | varchar   |             | Unique session identifier                   |
| predicted_sign | varchar   |             | Top-1 predicted gloss                       |
| confidence     | float     |             | Confidence score for the top prediction     |
| top_k          | array     |             | Top-5 predictions `[{sign, confidence}, …]` |
| timestamp      | timestamp |             | When the prediction was made                |
| feedback       | varchar   | NULL        | User feedback (`correct` / `wrong`)         |
