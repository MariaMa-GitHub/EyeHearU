## ML Pipeline (`ml/`) — ASL Video Classifier

This folder contains the **machine learning code** for Eye Hear U.
It defines the ASL video classifier, training loop, and evaluation scripts.

---

### Approach

We use **Approach B: Video Classifier** — a 3D CNN backbone (e.g., R3D-18, MC3-18, or R(2+1)D-18, pretrained on Kinetics-400) fine-tuned on short ASL sign video clips.

The classifier takes as input a tensor of shape `(B, 3, 16, 224, 224)` — a batch of 16-frame video clips resized to 224×224 — and outputs logits over the full gloss vocabulary (~2,000+ classes from ASL Citizen, supplemented by WLASL and MS-ASL).

---

### Key Files & Folders

- **`config.py`**
  Central configuration (dataclasses) for data paths, model backbone, training hyperparameters, and device selection.

- **`models/classifier.py`**
  `ASLVideoClassifier`: wraps a torchvision 3D CNN backbone (R3D-18, MC3-18, R(2+1)D-18) with a custom dropout + linear classification head.

- **`training/dataset.py`**
  `ASLVideoDataset`: reads preprocessed `.mp4` clips from `data/processed/clips/{split}/{gloss}/`, samples 16 frames, applies ImageNet normalisation and optional augmentations, and returns `(C, T, H, W)` tensors.

- **`training/train.py`**
  End-to-end training script: builds DataLoaders, instantiates the classifier, trains with AdamW + cosine LR schedule, freezes the backbone for the first N epochs, applies early stopping, and saves checkpoints.

- **`evaluation/evaluate.py`**
  Loads a checkpoint + test set, computes overall accuracy, top-5 accuracy, per-class accuracy, and lists the most confused sign pairs. Saves results to `evaluation_results.json`.

- **`checkpoints/`**
  Model weights: `best_model.pt` (loaded by the backend for inference).

---

### How to Train

1. **Run the data pipeline first** (see `docs/data_pipeline.md`):

   ```bash
   cd data/scripts
   python ingest_asl_citizen.py
   python ingest_wlasl.py
   python ingest_msasl.py
   python preprocess_clips.py
   python build_unified_dataset.py
   python validate.py
   ```

2. **Install ML dependencies:**

   ```bash
   cd ml
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Train:**

   ```bash
   python -m training.train
   ```

4. **Evaluate:**

   ```bash
   python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
   ```

---

### Who Works Here

- **Siyi & Chloe (ML):** Model architecture, hyperparameter tuning, confusion matrix analysis.
- **Zhixiao (Backend):** Uses `best_model.pt` and `label_map.json` for inference integration.
