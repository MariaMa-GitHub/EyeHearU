# Part Three: Data Processing Pipeline (WLASL)

This section describes the initial data processing pipeline we built for the WLASL dataset.  
The goal is to take raw sign videos and turn them into clean training data for our ASL classifier.

In our implementation, we download the WLASL metadata JSON from the official GitHub repository using `data/scripts/download_wlasl.py`. Videos are sourced from YouTube and other providers per the metadata; many links may be unavailable, so we recommend supplementing with pre-downloaded videos from the WLASL repo or custom recordings.

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

In the official WLASL dataset, videos are sourced from YouTube, ASLBrick, ASL SignBank, and other providers; many YouTube links may be unavailable.  
Our script downloads metadata from GitHub. Videos can be placed manually in `data/raw/wlasl/videos/` (see Code section).

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

### WLASL Raw Metadata JSON Structure

The file `data/raw/wlasl/WLASL_v0.3.json` (downloaded by `download_wlasl.py`) has this shape:

```json
[
  {
    "gloss": "hello",
    "instances": [
      {
        "video_id": "68011",
        "instance_id": 0,
        "split": "train",
        "signer_id": 110,
        "source": "valencia-asl",
        "url": "https://www.youtube.com/watch?v=...",
        "bbox": [x1, y1, x2, y2],
        "fps": 25,
        "frame_start": 1,
        "frame_end": -1
      }
    ]
  }
]
```

- `gloss` → our `gloss_id`
- `instances[].video_id` → matches local file `{video_id}.mp4` in `videos/`
- `instances[].split` → train/val/test
- `instances[].url` → original source (YouTube, ASLBrick, etc.)
- `instances[].signer_id`, `instances[].source` → available for stratification

---

**Note:**  
The full schema is shared with the other dataset, but for WLASL many optional fields are just stored as `null`.  
The raw WLASL metadata JSON includes `url` per instance (YouTube, ASLBrick, etc.); we use `video_id` to match local files in `videos/`.

---

## Pipeline Diagrams (Technologies Used)

### Offline Training Pipeline (WLASL)

```text
WLASL Metadata JSON (download_wlasl.py) + Raw Videos (data/raw/wlasl/videos/)
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
We use the `split` field (train/val/test) from the official WLASL metadata per instance.  
The label map is built from `ml/config.py` target vocab; only glosses present in WLASL are included.  
If we re-split later, we will regenerate split labels and document the ratios.

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
| Ingestion | Run `python data/scripts/download_wlasl.py` to fetch metadata; place videos in `data/raw/wlasl/videos/` manually |
| Cleaning + Transform | After raw videos are available |
| Splitting + Label Map | Produced by `download_wlasl.py` (filtered by target vocab in `ml/config.py`) |
| Retraining | When we add new samples or update the dataset |

---

### Main Use Case: Initial Training on WLASL

1. Run `download_wlasl.py` to download WLASL metadata JSON from GitHub  
2. Place video files in `data/raw/wlasl/videos/` (manual; see WLASL repo)  
3. Match metadata entries to locally available video files  
4. Extract a few frames per video  
5. Remove unusable frames  
6. Resize images (normalization is applied at training time)  
7. Split into train/val/test (per metadata; label map from target vocab)  
8. Train classifier model  

---

### Dataset Specific Issues (WLASL)

WLASL introduces some extra challenges:

- The original metadata includes many YouTube URLs; a subset may be broken or unavailable  
- We download metadata from GitHub; videos are placed manually (or via WLASL repo) in `data/raw/wlasl/videos/`  
- Signer metadata exists (`signer_id`), but is not reliable enough for strict signer-stratified splitting  
- Some target glosses (e.g. digits 1–10, letters c/l/x/y/z, "allergic") may not appear in WLASL; the script reports missing glosses  

These are handled by filtering to target vocab, validation, and leaving missing fields as null.

(We verify completeness by checking how many target glosses from `ml/config.py` are found in the WLASL JSON and how many video IDs are present locally.)

---

## Code for Initial Pipeline Version

Our initial pipeline implementation is organized like this:
```text
data/
├── scripts/
│   └── download_wlasl.py   # downloads metadata, filters to target vocab, extracts frames
├── raw/
│   └── wlasl/
│       ├── WLASL_v0.3.json   # official metadata (gloss + instances + split)
│       └── videos/           # place .mp4 files here (manual download from WLASL repo)
└── processed/
    ├── label_map.json       # gloss → int (filtered by ml/config.py target_vocab)
    └── images/
        └── train/           # extracted frames by gloss
```

Example command:
```bash
python data/scripts/download_wlasl.py
```

Prerequisites:
```bash
pip install opencv-python requests tqdm
```

The script:
1. Downloads `WLASL_v0.3.json` from GitHub (or uses existing file)
2. Filters to target vocab from `ml/config.py`, saves `label_map.json`
3. If `data/raw/wlasl/videos/` exists, extracts frames from matching videos (by `video_id`)

Planned stages (not fully implemented as a single runner script yet):
- ingest (metadata download done; video ingestion manual)
- clean  
- transform  
- load  
- report  

---

## Next Steps (Not Yet Implemented)

Some features are planned but not finished yet:

- Verifying dataset completeness between local video files in `data/raw/wlasl/videos/` and the official WLASL metadata JSON  
  (e.g., checking for any missing video IDs)

- Signer-stratified splitting  
  (WLASL signer metadata is incomplete, so we may need another approach)

- Cloud-based orchestration (AWS Step Functions)  
  Right now everything runs locally

- Adding custom recordings to fill low-sample glosses  
  (You should confirm which target signs are missing from WLASL)
