# Benchmarking and Evaluation

This document describes how to reproduce the evaluation benchmarks for Eye Hear U's ASL sign classifier, including accuracy metrics, per-class analysis, and inference latency.

## Evaluation Metrics

The I3D evaluation script (`ml/i3d_msft/evaluate.py`) computes:

| Metric | Description |
|--------|-------------|
| **Top-k Accuracy** | Fraction where the true label is among the top k predictions |
| **MRR** | Mean Reciprocal Rank of the correct prediction |
| **DCG** | Discounted Cumulative Gain of the correct prediction |
| **Confusion Matrix** | Full N x N matrix (saved as JSON) |

## Reproducing Results

### Prerequisites

```bash
cd ml
pip install -r requirements.txt
```

### Run Evaluation

```bash
cd ml
python -m i3d_msft.evaluate \
    --checkpoint path/to/best_model.pt \
    --split-csv path/to/test.csv \
    --clip-dir path/to/clips/ \
    --topk 1 5 10
```

Outputs are written to `ml/evaluation_results/`:
- `evaluation_results.json` — all numeric metrics
- `confusion_matrix.json` — full confusion matrix with class names

### Reproducibility

All evaluation runs use deterministic seeding:
- `torch.manual_seed(42)`, `numpy`, and `random` seeds are set before any computation
- DataLoader shuffling is disabled for evaluation

Training also sets deterministic seeds and uses `torch.backends.cudnn.deterministic = True`.

## Training

Training is performed via Modal cloud GPU using the I3D architecture. See [I3D S3 Repro Guide](i3d_s3_repro_guide.md) and [Ops Migration Tutorial](ops_migration_modal_sft_tutorial.md) for full details.

```bash
pip install modal
modal setup  # one-time auth

# Smoke test
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 1 --clip-limit 200

# Full training
modal run ml/modal_train_i3d.py --bucket eye-hear-u-public-data-ca1 --epochs 20
```

Key training features:
- **Head-only phase** followed by full backbone fine-tuning
- **S3 checkpoint uploads** during training for resilience against disconnects
- Best checkpoint saved and uploaded to S3

## Deployed Model (Inception I3D)

The production model is an Inception I3D fine-tuned on 856 ASL gloss classes (v4, candidate-ac-eval-v4). The deployed model:

- Accepts 64-frame video clips at 224x224 resolution
- Uses `[-1, 1]` pixel normalisation
- Outputs `(1, num_classes, T')` logits, temporally max-pooled to `(1, num_classes)`
- Achieves high accuracy on the 856-class vocabulary

## Inference Latency

For production benchmarking of the FastAPI endpoint:

```bash
# Start the backend
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000

# Benchmark with a sample video
time curl -X POST http://localhost:8000/api/v1/predict \
    -F "file=@sample_video.mp4"
```

Latency depends on hardware (CPU vs GPU), video length, and preprocessing overhead. The backend preprocesses uploaded videos with OpenCV before running model inference.
