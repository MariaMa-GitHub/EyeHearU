# Part Three: Data Processing Pipeline (WLASL)

This section describes the initial data processing pipeline we built for the WLASL dataset.  
The goal is to take raw sign videos and turn them into clean training data for our ASL classifier.

In our implementation, we use a Kaggle-hosted version of WLASL2000 where the video files are already stored locally, which helps avoid missing or unavailable URLs in the original release.

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

In the official WLASL dataset, videos are sourced from YouTube and many links may be unavailable.  
To avoid this issue, we use a Kaggle version where videos are already included locally in a `videos/` folder.

| Field | Description |
|------|------------|
| video_id | unique video identifier |
| gloss_id | label for this video |
| dataset_source | "wlasl" |
| local_path | where the video is stored locally |
| duration_sec | extracted from video |
| fps / resolution | extracted if available (may differ since videos are resized) |
| signer_id | available in metadata, but we do not fully rely on it (may be incomplete for our use cases) |

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
| normalized | optional (depends on training pipeline) |

---

**Note:**  
The full schema is shared with the other dataset, but for WLASL many optional fields are just stored as `null`.  
Some original metadata fields (such as YouTube URLs) are not used in the Kaggle version.

---

## Pipeline Diagrams (Technologies Used)

### Offline Training Pipeline (WLASL)

```text
WLASL Raw Videos (local Kaggle dataset)
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
We currently use the split field provided in the official WLASL metadata JSON.  
If we re-split the dataset later, we will regenerate split labels and document the ratios.

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
| Ingestion | When setting up the Kaggle WLASL dataset locally |
| Cleaning + Transform | After raw videos are available |
| Splitting + Label Map | Before training |
| Retraining | When we add new samples or update the dataset |

---

### Main Use Case: Initial Training on WLASL

1. Parse WLASL metadata JSON  
2. Match metadata entries to locally available video files  
3. Extract a few frames per video  
4. Remove unusable frames  
5. Resize images (normalization is applied at training time)
6. Split into train/val/test  
7. Train classifier model  

---

### Dataset Specific Issues (WLASL)

WLASL introduces some extra challenges:

- The original dataset contains broken YouTube links, so we rely on a Kaggle-hosted version with local videos  
- Signer metadata exists, but is not reliable enough for strict signer-stratified splitting
- Some gloss classes have very few usable samples  

These are handled by filtering, validation, and leaving missing fields as null.

(We plan to verify completeness by checking how many video IDs from the JSON are present in the local dataset.)

---

## Code for Initial Pipeline Version

Our initial pipeline implementation is organized like this:
```text
wlasl-complete/
find_missing.py        # checks whether all video IDs exist locally
WLASL_v0.3.json         # official metadata (gloss + instances + split)
wlasl_class_list.txt    # list of gloss classes
videos/                 # Kaggle-provided local video files
```

Example command:
```bash
python find_missing.py
```
This script checks whether video IDs listed in `WLASL_v0.3.json` exist in the local `videos/` folder.

Planned stages (not fully implemented as a single runner script yet):
- ingest  
- clean  
- transform  
- load  
- report  

---

## Next Steps (Not Yet Implemented)

Some features are planned but not finished yet:

- Verifying dataset completeness between the Kaggle video files and the official WLASL metadata JSON  
  (e.g., checking for any missing video IDs)

- Signer-stratified splitting  
  (WLASL signer metadata is incomplete, so we may need another approach)

- Cloud-based orchestration (AWS Step Functions)  
  Right now everything runs locally

- Adding custom recordings to fill low-sample glosses  
  (You should confirm which target signs are missing from WLASL)
