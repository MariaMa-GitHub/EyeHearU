# Evaluation Metrics — Step-by-Step Guide

How to generate accuracy, precision/recall/F1, confusion matrices, and inference latency numbers for your report.

---

## The Deployed Model

Eye Hear U uses **Microsoft's Inception I3D** for all production inference. This is the model that the mobile app and backend API use.

| Property | Value |
|----------|-------|
| **Architecture** | Inception I3D (`ml/i3d_msft/pytorch_i3d.py`) |
| **Input** | `(1, 3, 64, 224, 224)` — 64 RGB frames at 224x224 |
| **Normalization** | `[-1, 1]` pixel range |
| **Classes** | 856 ASL glosses |
| **Label map** | `ml/i3d_label_map_mvp-sft-full-v1.json` |
| **Checkpoint** | Auto-downloaded from S3 by the backend on first startup |
| **Evaluate via** | Backend API (`POST /api/v1/predict`) — see [Section 5](#5-evaluating-the-deployed-i3d-model-via-the-backend-api) |

The repo also contains an **in-repo baseline** (`ml/models/classifier.py`, torchvision R3D-18, 16 frames, ImageNet normalization) with its own training/evaluation scripts. This baseline is **not deployed** — it exists only for reproducible training experiments. Section 3-4 cover the baseline; Section 5 covers the deployed I3D model.

---

## 1. Prerequisites

### Install ML dependencies

```bash
cd ml
pip install -r requirements.txt
```

Key packages: `torch>=2.2.0`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `opencv-python`.

### Check your device

The default training device is `mps` (Apple Silicon). If you are on a different machine:

```bash
# Check what's available
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

If you need to change the device, edit `ml/config.py` line `device: str = "mps"` to `"cpu"` or `"cuda"`.

---

## 2. Preparing the Data

The evaluation script reads processed video clips from a specific directory structure. **You must run the data pipeline first.**

### Expected directory layout

```
data/processed/
├── label_map.json              ← {"gloss_name": class_index, ...}
└── clips/
    ├── train/
    │   ├── hello/
    │   │   ├── clip_001.mp4
    │   │   └── clip_002.mp4
    │   ├── thanks/
    │   └── ...
    ├── val/
    │   ├── hello/
    │   └── ...
    └── test/
        ├── hello/
        └── ...
```

- Each `.mp4` is a short video clip of a single ASL sign
- Folder names = gloss labels (must match keys in `label_map.json`)
- The dataset loader reads all `.mp4` files from each gloss subfolder

### Running the data pipeline

```bash
cd data/scripts
pip install -r requirements.txt

export PIPELINE_ENV=local
python ingest_asl_citizen.py      # download ASL Citizen dataset
python preprocess_clips.py        # resize/trim clips to standard format
python build_unified_dataset.py   # organize into train/val/test splits
python validate.py                # verify structure
```

After this, `data/processed/clips/` and `data/processed/label_map.json` should exist.

### Verify the data is ready

```bash
# Check label_map exists and count classes
python3 -c "
import json
lm = json.load(open('../data/processed/label_map.json'))
print(f'{len(lm)} classes')
print('First 5:', list(lm.keys())[:5])
"

# Check clip counts per split
for split in train val test; do
    count=$(find ../data/processed/clips/$split -name '*.mp4' 2>/dev/null | wc -l)
    echo "$split: $count clips"
done
```

---

## 3. Training the In-Repo Baseline (R3D-18)

**Where to run:** from the `ml/` directory.

```bash
cd ml
python -m training.train
```

### What happens

1. Reads config from `ml/config.py` (batch_size=8, lr=1e-3, 30 epochs, etc.)
2. Loads data from `data/processed/clips/{train,val}/`
3. Creates an R3D-18 model with a classification head for N classes
4. Trains with:
   - Backbone frozen for the first 3 epochs (transfer learning warmup)
   - Cosine annealing LR schedule
   - Early stopping (patience=5 epochs)
5. Saves checkpoints to `ml/checkpoints/`

### Where checkpoints go

```
ml/checkpoints/
├── best_model.pt       ← saved whenever val accuracy improves (this is the one you evaluate)
├── epoch_5.pt          ← periodic checkpoint
├── epoch_10.pt
└── ...
```

### Training output (example)

```
Loaded label map: 856 classes
Device: mps
Model parameters: 33,456,856

Epoch   1/30 | Train Loss: 3.12  Acc: 0.05 | Val Loss: 2.89  Acc: 0.09 | LR: 0.001000 | 45.2s
Epoch   2/30 | Train Loss: 2.45  Acc: 0.18 | Val Loss: 2.10  Acc: 0.23 | LR: 0.000998 | 44.8s
  -> New best (val_acc=0.2301)
...
Early stopping at epoch 18

Done. Best validation accuracy: 0.4512
```

---

## 4. Running the Evaluation Script

### The command

```bash
cd ml
python -m evaluation.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --seed 42
```

**All three arguments are optional:**

| Flag | Default | What it does |
|------|---------|--------------|
| `--checkpoint` | `checkpoints/best_model.pt` | Path to the `.pt` state_dict file |
| `--split` | `test` | Which data split to evaluate: `train`, `val`, or `test` |
| `--seed` | `42` | Random seed for reproducibility |

### What it computes

| Metric | What it means | Why it matters for your report |
|--------|---------------|-------------------------------|
| **Overall Accuracy (top-1)** | % of clips where `argmax(logits) == true_label` | Primary comparison metric in ASL recognition papers |
| **Top-5 Accuracy** | % of clips where true label is in the 5 highest-scoring predictions | Shows model "nearly knows" the sign even when top-1 misses |
| **Macro Precision** | Average precision across all classes (unweighted) | Measures false positive rate, class-balanced |
| **Macro Recall** | Average recall across all classes (unweighted) | Measures false negative rate, class-balanced |
| **Macro F1** | Harmonic mean of macro precision and recall | Single balanced metric for multi-class performance |
| **Per-class accuracy/P/R/F1** | Metrics broken down by individual sign | Identifies which signs are easy vs. hard |
| **Confusion matrix** | N x N table: `matrix[true][predicted] = count` | Reveals systematic confusions (e.g., "hello" vs. "goodbye") |
| **Top confusion pairs** | The 20 most frequent misclassification pairs | Highlights for error analysis section of report |
| **Inference latency** | Per-sample time in ms: mean, p50, p95, p99 | Proves real-time capability for mobile deployment |

### Output files

After running, check `ml/evaluation_results/`:

```
ml/evaluation_results/
├── evaluation_results.json     ← all metrics as JSON (parseable for tables)
├── confusion_matrix.json       ← full N x N matrix with class names
└── confusion_matrix.png        ← heatmap visualization (if matplotlib installed)
```

### Console output (example)

```
============================================================
Evaluation — test set  (seed=42)
============================================================
Overall Accuracy:  0.4512
Top-5 Accuracy:    0.8234
Macro Precision:   0.3901
Macro Recall:      0.3745
Macro F1:          0.3651
Total Samples:     8320

Inference Latency (per sample):
  Mean:  12.3 ms
  p50:   11.8 ms
  p95:   18.4 ms
  p99:   24.1 ms

Per-class detail:
  about               : acc=0.600  P=0.550  R=0.600  F1=0.574
  absent              : acc=0.450  P=0.400  R=0.450  F1=0.424
  ...

Most confused pairs:
  hello           -> goodbye         (15)
  please          -> thank_you       (12)
  ...

Results saved to evaluation_results/evaluation_results.json
Confusion matrix JSON saved to evaluation_results/confusion_matrix.json
Confusion matrix plot saved to evaluation_results/confusion_matrix.png
```

### Evaluating on different splits

Run on validation set (to compare with training output):

```bash
python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt --split val
```

Run on training set (to check for overfitting — train accuracy should be higher):

```bash
python -m evaluation.evaluate --checkpoint checkpoints/best_model.pt --split train
```

---

## 5. Evaluating the Deployed I3D Model via the Backend API

The production model (Inception I3D, 856 classes, 64-frame input) is a different architecture from the in-repo baseline. It is loaded and served by the backend API. You can evaluate it by sending test videos to the API endpoint.

### Start the backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # if not already done

# The first run downloads the model from S3 (~50 MB) to backend/model_cache/
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The backend automatically:
1. Downloads `best_model.pt` from S3 if not cached locally in `backend/model_cache/`
2. Loads `ml/i3d_label_map_mvp-sft-full-v1.json` (856 classes)
3. Preprocesses uploaded videos (decode, resize, normalize to [-1,1], 64 frames, center-crop 224x224)

### Send a test video

```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
    -F "file=@path/to/test_video.mp4"

# Example response:
# {
#   "sign": "hello",
#   "confidence": 0.92,
#   "top_k": [
#     {"sign": "hello", "confidence": 0.92},
#     {"sign": "hi", "confidence": 0.04},
#     {"sign": "hey", "confidence": 0.02}
#   ]
# }
```

### Batch evaluation with a script

To evaluate the I3D model on a set of test videos, you can write a simple script:

```python
"""Evaluate the deployed I3D model via the backend API on a folder of test videos."""
import json, requests, sys
from pathlib import Path
from collections import defaultdict

API_URL = "http://localhost:8000/api/v1/predict"

def evaluate_folder(test_dir: str):
    """
    Expects: test_dir/{gloss_name}/*.mp4
    Prints top-1 and top-5 accuracy.
    """
    test_path = Path(test_dir)
    correct_top1, correct_top5, total = 0, 0, 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for gloss_dir in sorted(test_path.iterdir()):
        if not gloss_dir.is_dir():
            continue
        true_label = gloss_dir.name.lower()

        for video in sorted(gloss_dir.glob("*.mp4")):
            with open(video, "rb") as f:
                resp = requests.post(API_URL, files={"file": f})

            if resp.status_code != 200:
                print(f"  SKIP {video.name}: {resp.status_code}")
                continue

            data = resp.json()
            top1 = data["sign"].lower()
            top5 = [p["sign"].lower() for p in data.get("top_k", [])]

            total += 1
            per_class[true_label]["total"] += 1

            if top1 == true_label:
                correct_top1 += 1
                per_class[true_label]["correct"] += 1
            if true_label in top5:
                correct_top5 += 1

    print(f"\nResults ({total} samples)")
    print(f"  Top-1 Accuracy: {correct_top1/max(total,1):.4f}")
    print(f"  Top-5 Accuracy: {correct_top5/max(total,1):.4f}")

    print(f"\nPer-class accuracy:")
    for name in sorted(per_class):
        d = per_class[name]
        acc = d['correct'] / max(d['total'], 1)
        print(f"  {name:20s}: {acc:.3f} ({d['correct']}/{d['total']})")

if __name__ == "__main__":
    evaluate_folder(sys.argv[1])
```

Run it:

```bash
# Assuming test clips are in data/processed/clips/test/
python evaluate_api.py ../data/processed/clips/test/
```

### Measuring API inference latency

```bash
# Time a single request end-to-end (includes network + preprocessing + inference)
time curl -s -X POST http://localhost:8000/api/v1/predict \
    -F "file=@test_video.mp4" -o /dev/null

# Or use the health endpoint to verify the API is running
curl http://localhost:8000/health
```

---

## 6. Extracting Numbers for Your Report

### From `evaluation_results.json`

After running the evaluation script, parse the JSON for your report tables:

```python
import json

with open("evaluation_results/evaluation_results.json") as f:
    r = json.load(f)

# ---- Table 1: Overall Metrics ----
print("| Metric | Value |")
print("|--------|-------|")
print(f"| Top-1 Accuracy | {r['overall_accuracy']:.4f} |")
print(f"| Top-5 Accuracy | {r['top5_accuracy']:.4f} |")
print(f"| Macro Precision | {r['macro_precision']:.4f} |")
print(f"| Macro Recall | {r['macro_recall']:.4f} |")
print(f"| Macro F1 | {r['macro_f1']:.4f} |")
print(f"| Total Samples | {r['total_samples']} |")

# ---- Table 2: Inference Latency ----
lat = r["inference_latency"]
print("\n| Latency | ms |")
print("|---------|-----|")
for k, v in lat.items():
    print(f"| {k} | {v:.1f} |")

# ---- Table 3: Most Confused Pairs ----
print("\n| True Sign | Predicted As | Count |")
print("|-----------|-------------|-------|")
for item in r["top_confusions"][:10]:
    print(f"| {item['true']} | {item['predicted']} | {item['count']} |")
```

### Confusion matrix heatmap

The script auto-generates `confusion_matrix.png`. For a report with 856 classes, the full matrix will be very dense. For a readable figure, you can plot a subset:

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open("evaluation_results/confusion_matrix.json") as f:
    data = json.load(f)

cm = np.array(data["matrix"])
names = data["class_names"]

# Pick the top 20 most frequent classes
class_totals = cm.sum(axis=1)
top_indices = np.argsort(-class_totals)[:20]

cm_sub = cm[np.ix_(top_indices, top_indices)]
names_sub = [names[i] for i in top_indices]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm_sub, xticklabels=names_sub, yticklabels=names_sub,
            cmap="Blues", annot=True, fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Top 20 Classes by Frequency)")
plt.tight_layout()
fig.savefig("evaluation_results/confusion_matrix_top20.png", dpi=150)
plt.show()
```

---

## 7. What Metrics to Include in Your Report

### Experiments section

| What to report | Where to get it | Why |
|-----------------|-----------------|-----|
| Top-1 / Top-5 accuracy on test set | `evaluation_results.json` | Primary result; compare with ASL Citizen paper baselines |
| Macro F1 on test set | `evaluation_results.json` | Balanced metric across all classes |
| Per-class accuracy (best/worst 10) | `per_class_detail` in JSON | Shows which signs are easy/hard |
| Confusion matrix (top 20 classes) | `confusion_matrix.png` or custom plot | Visual evidence of model behavior |
| Top confusion pairs | `top_confusions` in JSON | Error analysis — what does the model struggle with? |
| Inference latency (p50, p95) | `inference_latency` in JSON | Proves real-time mobile deployment is feasible |
| Train vs. val accuracy curve | Training console output (epoch logs) | Shows convergence and overfitting behavior |

### Comparison with published baselines

The ASL Citizen paper reports results on their test set. If you used the same data splits, you can directly compare:

| Model | Top-1 Acc | Top-5 Acc | Source |
|-------|-----------|-----------|--------|
| I3D (ASL Citizen paper) | — | — | Li et al. 2024 |
| Our I3D (856 classes) | — | — | This project (deployed) |
| Our R3D-18 baseline | — | — | This project (in-repo) |

Fill in from your evaluation results.

---

## Troubleshooting

### "No training data found"

```
ERROR: No training data found. Run the data pipeline first.
  Expected clips at: ../data/processed/clips/train/
```

The data pipeline has not been run. See [Section 2](#2-preparing-the-data).

### "FileNotFoundError: checkpoints/best_model.pt"

You need to train the model first (`python -m training.train`). The checkpoint is created during training. See [Section 3](#3-training-the-in-repo-baseline-r3d-18).

### "label_map.json not found"

The data pipeline creates this file. Make sure `data/processed/label_map.json` exists after running the pipeline.

### evaluation_results/ already has files from before

The existing `evaluation_results.json` in the repo contains **dummy data from test runs** (2 classes, 9 samples). It will be overwritten when you run a real evaluation. You can verify by checking the `total_samples` field — real results will have thousands of samples.

### Device errors (MPS/CUDA)

If you get device-related errors, edit `ml/config.py` and set `device: str = "cpu"`. CPU is slower but always works.
