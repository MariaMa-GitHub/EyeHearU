# Benchmarking and Evaluation

This document describes how to reproduce the evaluation benchmarks for Eye Hear U's ASL sign classifier, including accuracy metrics, per-class analysis, and inference latency.

## Evaluation Metrics

The evaluation script (`ml/evaluation/evaluate.py`) computes:

| Metric | Description |
|--------|-------------|
| **Overall Accuracy** | Fraction of correctly classified samples (top-1) |
| **Top-5 Accuracy** | Fraction where the true label is among the top 5 predictions |
| **Macro Precision** | Unweighted mean of per-class precision |
| **Macro Recall** | Unweighted mean of per-class recall |
| **Macro F1** | Unweighted mean of per-class F1 scores |
| **Per-class Detail** | Accuracy, precision, recall, and F1 for each gloss |
| **Confusion Matrix** | Full N x N matrix (saved as JSON and PNG heatmap) |
| **Top Confusion Pairs** | The 20 most common misclassification pairs |
| **Inference Latency** | Per-sample latency: mean, p50, p95, p99 (milliseconds) |

## Reproducing Results

### Prerequisites

```bash
cd ml
pip install -r requirements.txt
```

Ensure the data pipeline has been run and produced:
- `data/processed/clips/{train,val,test}/{gloss}/*.mp4`
- `data/processed/label_map.json`

### Run Evaluation

```bash
cd ml
python -m evaluation.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --seed 42
```

Outputs are written to `ml/evaluation_results/`:
- `evaluation_results.json` — all numeric metrics
- `confusion_matrix.json` — full confusion matrix with class names
- `confusion_matrix.png` — heatmap visualisation

### Reproducibility

All evaluation runs use deterministic seeding:
- `torch.manual_seed(42)`, `numpy`, and `random` seeds are set before any computation
- DataLoader shuffling is disabled for evaluation
- Pass `--seed <value>` to use a different seed

Training also sets deterministic seeds and uses `torch.backends.cudnn.deterministic = True`.

## Training Benchmarks

The training script logs per-epoch metrics:

```
Epoch   1/30 | Train Loss: 3.1245  Acc: 0.0523 | Val Loss: 2.8901  Acc: 0.0912 | LR: 0.001000 | 45.2s
Epoch   2/30 | Train Loss: 2.4501  Acc: 0.1834 | Val Loss: 2.1023  Acc: 0.2301 | LR: 0.000998 | 44.8s
...
```

Key training features:
- **Early stopping** with patience of 5 epochs on validation accuracy
- **Cosine annealing** learning rate schedule
- **Backbone freezing** for the first N epochs (transfer learning)
- Best checkpoint saved to `checkpoints/best_model.pt`

### Run Training

```bash
cd ml
python -m training.train
```

Training configuration is in `ml/config.py`. Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone` | `r3d_18` | 3D CNN architecture (r3d_18, mc3_18, r2plus1d_18) |
| `num_frames` | 16 | Frames sampled per clip |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `epochs` | 30 | Maximum training epochs |
| `early_stopping_patience` | 5 | Epochs without improvement before stopping |
| `backbone_freeze_epochs` | 3 | Epochs with frozen backbone weights |

## Deployed Model (Inception I3D)

The production model is an Inception I3D fine-tuned on the MVP gloss set (48 classes). Training was performed on the `freya-a5-training` branch using Microsoft's ASL-Citizen I3D codebase. The deployed model:

- Accepts 64-frame video clips at 224x224 resolution
- Uses `[-1, 1]` pixel normalisation (different from the in-repo R3D baseline which uses ImageNet normalisation)
- Outputs `(1, num_classes, T')` logits, temporally max-pooled to `(1, num_classes)`
- Achieves high accuracy on the 48-class MVP vocabulary

The in-repo R3D baseline (`ml/models/classifier.py`) serves as a reproducible training benchmark using torchvision's 3D CNNs with 16-frame clips and ImageNet normalisation.

## Inference Latency

The evaluation script measures per-sample inference latency during the evaluation pass. For production benchmarking of the FastAPI endpoint:

```bash
# Start the backend
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# Benchmark with a sample video
time curl -X POST http://localhost:8000/api/v1/predict \
    -F "file=@sample_video.mp4"
```

Latency depends on hardware (CPU vs GPU), video length, and preprocessing overhead. The backend preprocesses uploaded videos with OpenCV before running model inference.
