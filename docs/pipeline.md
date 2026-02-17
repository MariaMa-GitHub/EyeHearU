# Part Three: Data Processing Pipeline (WLASL)

This section describes the initial data processing pipeline we built for the WLASL dataset.  
The goal is to take raw sign videos and turn them into clean training data for our ASL classifier.

Some fields in our shared schema are not available in WLASL (for example signer consent info), so those values are stored as `null` for now.

---

## Data Schemas

We use one shared schema across datasets, but here we focus on the parts that apply to WLASL.

### Key Tables / Entities for WLASL

**sign_glosses**  
Stores the vocabulary of signs.

WLASL provides gloss labels, but does not always provide extra metadata like difficulty or category, so those may be left as null.

Example fields:

| Field | Description |
|------|------------|
| gloss_id | sign label (e.g. “hello”) |
| gloss_name | readable name |
| category | null (not provided in WLASL) |
| difficulty | null (not provided in WLASL) |
| num_samples | number of usable videos |

---

**sign_videos**  
Stores raw video metadata.

WLASL videos are mostly sourced from YouTube, so some links may be broken.

| Field | Description |
|------|------------|
| video_id | unique video identifier |
| gloss_id | label for this video |
| dataset_source | "wlasl" |
| s3_path / local_path | where the video is stored |
| duration_sec | extracted from video |
| fps / resolution | extracted if available |
| signer_id | null (WLASL does not reliably provide signer info) |

---

**extracted_frames**  
Frames extracted from videos for training.

| Field | Description |
|------|------------|
| frame_id | unique frame identifier |
| video_id | source video |
| gloss_id | inherited label |
| frame_index | position in the video |
| is_blurry / is_black | quality flags |
| hand_detected | whether MediaPipe finds a hand |

---

**processed_images**  
Final training ready images.

| Field | Description |
|------|------------|
| image_id | processed sample id |
| gloss_id | training label |
| split | train / val / test |
| file_path | stored output |
| width, height | always 224 × 224 |
| normalized | true |

---

**Note:**  
The full schema is shared with the other dataset, but for WLASL many optional fields are just stored as `null`.  
(You can reference the schema doc in our repo.)

---

## Pipeline Diagrams (Technologies Used)

### Offline Training Pipeline (WLASL)
```text
WLASL Raw Videos
      ↓
Frame Extraction (OpenCV)
      ↓
Cleaning (remove black/blurry frames, dedup)
      ↓
Transformation (hand crop, resize to 224×224)
      ↓
Train/Val/Test Split + Label Map
      ↓
Model Training (PyTorch)
```

### Tools Used

- Python scripts for preprocessing
- OpenCV for video + frame extraction
- MediaPipe for hand region detection
- PyTorch dataset loader for training

AWS services are listed in the full pipeline design, but our current implementation is local-first.

---

## When the Pipelines Run and Use Cases

### When it runs

For WLASL, the pipeline is not running continuously.  
It is triggered manually when we need to prepare data for training.

| Pipeline Stage | Trigger |
|--------------|---------|
| Ingestion | When downloading WLASL release |
| Cleaning + Transform | After raw videos are available |
| Splitting + Label Map | Before training |
| Retraining | When we add new samples or fix broken links |

---

### Main Use Case: Initial Training on WLASL

1. Parse WLASL metadata JSON  
2. Download available videos  
3. Extract a few frames per video  
4. Remove unusable frames  
5. Resize and normalize images  
6. Split into train/val/test  
7. Train classifier model

---

### Dataset Specific Issues (WLASL)

WLASL introduces some extra challenges:

- Many videos come from YouTube and may be missing
- Signer identity is not consistently available
- Some gloss classes have very few usable samples

These are handled by filtering and leaving missing fields as null.

(You should double check how many broken links you actually see in your local download.)

---

## Code for Initial Pipeline Version

Our initial pipeline implementation is organized like this:
```text
data/scripts/
download_wlasl.py # download + metadata parsing
pipeline.py # main stage runner
preprocess.py # hand crop + resize utilities
```

Example command:
```bash
python pipeline.py --stage all
```

Stages include:

- ingest  
- clean  
- transform  
- load  
- report  

This matches the structure described in the pipeline writeup.

---

## Next Steps (Not Yet Implemented)

Some features are planned but not finished yet:

- Automatically handling missing/broken WLASL video links  
  (I need to verify the percentage of unavailable videos)

- Signer-stratified splitting  
  (WLASL signer metadata is incomplete, so we may need another approach)

- Cloud-based orchestration (AWS Step Functions)  
  Right now everything runs locally

- Adding custom recordings to fill low-sample glosses  
  (You should confirm which target signs are missing from WLASL)
