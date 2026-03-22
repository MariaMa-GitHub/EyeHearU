## ML (`ml/`) — training code and inference artifacts

This directory holds **training scripts**, **shared config**, the **Inception I3D** module used at inference, and **label maps** for the MVP checkpoint.

---

### Inference (used by the FastAPI backend)

- **`i3d_msft/`** — `pytorch_i3d.py`, `videotransforms.py` (must stay aligned with the training branch that produced the checkpoint).
- **`i3d_label_map_mvp-sft-full-v1.json`** — class index ↔ gloss for the MVP model.

The backend decodes uploads with **`backend/app/services/preprocessing.py`**, documented in **`docs/PREPROCESSING.md`**.

---

### Training (in-repo baseline)

- **`config.py`** — dataclass configuration for paths, backbone, hyperparameters.
- **`models/classifier.py`** — `ASLVideoClassifier` around a torchvision **3D CNN** (e.g. R3D-18), **16-frame** clips, `(B, 3, 16, 224, 224)`.
- **`training/dataset.py`** / **`training/train.py`** / **`evaluation/evaluate.py`** — standard train/eval loop on processed clips under `data/processed/`.

The **deployed MVP API** uses the **I3D + 64-frame** pipeline, not this R3D baseline, unless the checkpoint and preprocessing are switched together.

---

### Train (after data pipeline)

1. Run the data pipeline (see `docs/data_pipeline.md` and `data/scripts/`).
2. Install dependencies:

   ```bash
   cd ml
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Train:

   ```bash
   python -m training.train
   ```

4. Evaluate:

   ```bash
   python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt
   ```

---

### Label maps

`i3d_msft/export_label_map.py` can help produce JSON maps compatible with the backend. The MVP map file is checked in at the repo root of `ml/`.

---

### Sync with the I3D training branch

Training for the shipped I3D checkpoint may live on a separate branch (see **`docs/DEVELOPER_GUIDE.md`**). After any change to **`i3d_msft/`** or training preprocessing, mirror updates in **`backend/app/services/preprocessing.py`** and run backend tests with **`--cov-fail-under=100`**.
