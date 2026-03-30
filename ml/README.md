## ML (`ml/`) — training code and inference artifacts

This directory holds **training scripts**, **shared config**, the **Inception I3D** module used at inference, **label maps** for the MVP checkpoint, and the **I3D training pipeline** with S3-based data management.

---

### Inference (used by the FastAPI backend)

- **`i3d_msft/pytorch_i3d.py`** — InceptionI3d architecture (from Microsoft).
- **`i3d_msft/videotransforms.py`** — RandomCrop, CenterCrop, RandomHorizontalFlip for video tensors.
- **`i3d_label_map_mvp-sft-full-v1.json`** — class index ↔ gloss for the MVP model (856 classes).

The backend decodes uploads with **`backend/app/services/preprocessing.py`**, documented in **`docs/PREPROCESSING.md`**.

---

### I3D Training (deployed model)

The training pipeline pulls data from S3 using versioned split plans and trains on Modal GPUs.

| File | Purpose |
|------|---------|
| `i3d_msft/train.py` | I3D training with S3 data, differential LR, backbone freezing |
| `i3d_msft/evaluate.py` | Evaluation: top-k accuracy, MRR, DCG, confusion matrix |
| `i3d_msft/dataset.py` | `ASLCitizenI3DDataset` — 64-frame clips, [-1,1] normalization |
| `i3d_msft/s3_data.py` | S3 sync helpers: split downloads, clip downloads |
| `i3d_msft/build_label_map_artifacts.py` | Rebuild label map from training splits |
| `i3d_msft/export_label_map.py` | CSV → JSON label map utility |
| `modal_train_i3d.py` | Modal GPU wrapper for cloud training |

**Quick start (Modal):**

```bash
pip install modal && modal setup
# Smoke test
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 1 --clip-limit 200
# Full training
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 20
```

See **`docs/I3D_TRAINING_S3_REPRODUCTION.md`** for the full reproducible workflow and **`docs/MODAL_AWS_SFT_MIGRATION.md`** for the AWS / Modal / SFT migration playbook.

---

### In-Repo Baseline (not deployed)

These files wrap torchvision 3D CNNs (R3D-18, MC3-18, R2Plus1D-18) for reproducible local experiments. The deployed MVP uses I3D, not this baseline.

- **`config.py`** — dataclass configuration for paths, backbone, hyperparameters.
- **`models/classifier.py`** — `ASLVideoClassifier`, **16-frame** clips, `(B, 3, 16, 224, 224)`, ImageNet normalization.
- **`training/dataset.py`** / **`training/train.py`** / **`evaluation/evaluate.py`** — standard train/eval loop on processed clips under `data/processed/`.

```bash
cd ml
pip install -r requirements.txt
python -m training.train
python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
```

---

### Label maps

`i3d_msft/export_label_map.py` produces JSON maps compatible with the backend. `i3d_msft/build_label_map_artifacts.py` rebuilds the exact label map from a training run's filtered splits. The MVP map file is checked in at `ml/i3d_label_map_mvp-sft-full-v1.json`.

---

### Testing

~194 tests with **100%** line and branch coverage on `i3d_msft/` and `modal_train_i3d.py` (same gate as CI):

```bash
cd ml && python3 -m pytest tests/ -v \
  --cov=i3d_msft --cov=modal_train_i3d \
  --cov-config=.coveragerc \
  --cov-fail-under=100
```

After any change to **`i3d_msft/`** or training preprocessing, mirror updates in **`backend/app/services/preprocessing.py`** and run backend tests with **`--cov-fail-under=100`**.
