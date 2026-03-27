# Profiling — Execution Time Analysis

## Overview

We profiled **5 key functions** using Python's [`cProfile`](https://docs.python.org/3/library/profile.html) module to identify bottlenecks and suggest optimizations. The profiling script lives at `ml/profiling/profile_functions.py` and can be re-run at any time.

**How to run:**

```bash
cd ml/
python -m profiling.profile_functions
```

Binary `.prof` files are saved to `ml/profiling/results/` for deeper analysis with `pstats` or visualization tools like `snakeviz`.

---

## Summary Table

| # | Function | Location | Wall Time | Total Calls | Bottleneck |
|---|----------|----------|-----------|-------------|------------|
| 1 | `preprocess_video` | `backend/app/services/preprocessing.py:183` | 0.93 s (3 calls) | 2,726 | `cv2.resize` (77% of time) |
| 2 | `predict` | `backend/app/services/model_service.py:125` | 4.11 s (3 calls) | 9,353 | `torch.conv3d` (60%) + `torch.max_pool3d` (30%) |
| 3 | `i3d_evaluate` | `ml/i3d_msft/evaluate.py:152` | 6.17 s (1 call, 20 samples) | 15,121 | `torch.conv3d` (96% — forward pass dominates) |
| 4 | `i3d_train_one_epoch` | `ml/i3d_msft/train.py:57` | 20.53 s (1 epoch, 20 samples) | 17,895 | `run_backward` (66%) + `torch.conv3d` (30%) |
| 5 | `build_gloss_dict_from_csv` | `ml/i3d_msft/export_label_map.py:25` | 0.75 s (50 calls × 5K rows) | 2,752,771 | `csv.DictReader.__next__` (56%) |

---

## Detailed Analysis

### 1. `preprocess_video` — Video Preprocessing Pipeline

**What it does:** Decodes an uploaded MP4, applies adaptive frame skip, resizes to short-side-256, pads/trims to 64 frames, center-crops to 224×224, and returns a `(1, 3, 64, 224, 224)` tensor.

**Profile (3 calls on 90-frame 480×640 video):**

```
ncalls  tottime  cumtime  function
   192    0.716    0.716  cv2.resize              ← 77% of total time
   192    0.050    0.050  cv2.VideoCapture.read    ← 5.4%
     3    0.021    0.021  numpy.asarray            ← 2.3%
   192    0.019    0.019  ndarray.astype           ← 2.0%
```

**Bottleneck:** `cv2.resize` is called **twice per frame** (once for mobile downscale, once for short-side-256 resize), totalling 192 resize calls across 3 videos. Each resize operates on full BGR uint8 frames.

**Optimization opportunities:**

1. **Combine the two resize steps into one.** Currently, `_load_rgb_frames` first scales 4K frames down to ≤1280px (mobile guard), then separately scales to short-side-256. These could be computed as a single scale factor, halving the number of `cv2.resize` calls. Estimated savings: ~35% of wall time.

2. **Decode at reduced resolution.** OpenCV's `VideoCapture` decodes full-res frames before we downscale. Using `ffmpeg` with `-vf scale=...` or hardware-accelerated decoding (e.g., `VideoCapture` with `CAP_PROP_HW_ACCELERATION`) could decode directly at target resolution.

3. **Batch the BGR→RGB conversion.** Currently `cv2.cvtColor` is called per-frame. After stacking into a numpy array, a single `arr[..., ::-1]` slice reverses channels in-place without a function call per frame.

---

### 2. `predict` — Model Inference (I3D)

**What it does:** Runs a single forward pass through InceptionI3d, max-pools over the temporal dimension, applies softmax, and returns top-k predictions.

**Profile (3 calls on (1, 3, 64, 224, 224) tensor, CPU):**

```
ncalls  tottime  cumtime  function
   174    2.456    2.456  torch.conv3d             ← 60% of total time
    39    1.234    1.234  torch.max_pool3d         ← 30%
   213    0.155    0.155  torch._C._nn.pad         ← 3.8%
   171    0.072    0.072  torch.batch_norm          ← 1.8%
```

**Bottleneck:** 3D convolutions dominate (174 conv3d calls across Inception modules). `max_pool3d` is the second-largest contributor.

**Optimization opportunities:**

1. **Use `torch.jit.trace` or `torch.compile`.** JIT compilation can fuse conv3d + batch_norm + relu sequences, reducing kernel launch overhead. On CPU, `torch.compile(mode="reduce-overhead")` can provide 10–30% speedup.

2. **Reduce input resolution.** The model takes 224×224 frames but was originally designed for Kinetics-400 at similar resolution. For latency-sensitive mobile inference, testing 160×160 or 192×192 inputs could significantly reduce conv3d compute while maintaining acceptable accuracy.

3. **Use half-precision (FP16).** On GPU deployments, `model.half()` with `torch.cuda.amp` would roughly halve conv3d time. On CPU, `torch.bfloat16` is available on modern processors.

4. **ONNX Runtime.** Exporting to ONNX and running with `onnxruntime` (with MLAS/oneDNN optimizations) can yield 1.5–3× speedup on CPU inference compared to vanilla PyTorch.

---

### 3. `i3d_evaluate` — I3D Evaluation Loop

**What it does:** Runs I3D inference on all samples in a DataLoader, collects predictions, computes top-k accuracy, MRR, DCG, and builds a confusion matrix.

**Profile (1 call, 20 samples in 5 batches of 4, InceptionI3d):**

```
ncalls  tottime  cumtime  function
   100    5.942    5.942  torch.conv3d             ← 96%
   100    0.095    0.095  torch.batch_norm          ← 1.5%
    85    0.028    0.028  torch.relu_               ← 0.5%
```

**Bottleneck:** Virtually all time is spent in the model's forward pass (conv3d). Post-processing (metric computation) is negligible.

**Optimization opportunities:**

1. **All inference optimizations from `predict` (above) apply.** JIT tracing, ONNX, resolution reduction, and FP16 would proportionally speed up evaluation.

2. **Increase batch size.** Larger batches amortize per-batch overhead (data transfer, kernel launch). On GPU, batch sizes of 16–32 are typical for evaluation.

3. **Use `torch.inference_mode()` instead of `@torch.no_grad()`.** `inference_mode` is slightly faster as it disables autograd dispatch entirely rather than just disabling gradient recording.

---

### 4. `i3d_train_one_epoch` — Single I3D Training Epoch

**What it does:** Full I3D forward + backward pass over all training batches, with loss computation and optimizer step.

**Profile (1 epoch, 20 samples in 5 batches of 4, InceptionI3d, CPU):**

```
ncalls  tottime  cumtime  function
     5   13.586   13.586  run_backward              ← 66%
   100    6.152    6.152  torch.conv3d              ← 30%
   100    0.436    0.436  torch.batch_norm           ← 2.1%
     5    0.000    0.254  optimizer.step             ← 1.2%
```

**Bottleneck:** Backpropagation (`run_backward`) consumes 2× more time than the forward pass because it must compute gradients through all 3D conv layers plus store intermediate activations.

**Optimization opportunities:**

1. **Mixed-precision training (AMP).** `torch.cuda.amp.autocast()` with `GradScaler` can halve forward/backward time on GPU. On CPU, `torch.amp.autocast("cpu", dtype=torch.bfloat16)` provides similar benefits on modern hardware.

2. **Gradient accumulation.** Instead of calling `optimizer.step()` every batch, accumulate gradients over 2–4 batches. This lets you use smaller batch sizes (less memory) while maintaining effective batch size.

3. **Freeze early layers.** The profiling shows backbone conv3d dominates. Freezing the first 2–3 stages of R3D-18 (or I3D) during fine-tuning reduces backward-pass computation. The existing `freeze_backbone()` method already supports this — the `backbone_freeze_epochs` config parameter controls when unfreezing occurs.

4. **Use `torch.compile`.** PyTorch 2.x's compile can fuse operations and optimize memory layout, yielding 10–40% speedup on training loops.

---

### 5. `build_gloss_dict_from_csv` — Label Map Construction

**What it does:** Reads a training CSV, extracts unique glosses (lowercased, stripped), sorts them alphabetically, and returns a `{gloss: index}` dictionary.

**Profile (50 calls × 5,000-row CSV):**

```
ncalls    tottime  cumtime  function
250050    0.254    0.422   csv.DictReader.__next__  ← 56%
500000    0.051    0.051   dict.get                 ← 6.8%
500000    0.050    0.050   str.strip                ← 6.7%
500002    0.035    0.035   builtins.len             ← 4.7%
250000    0.028    0.028   str.lower                ← 3.7%
```

**Bottleneck:** `csv.DictReader` is slow because it creates a new dictionary for every row. With 5,000 rows × 50 calls = 250,000 row-parsing operations, this adds up.

**Optimization opportunities:**

1. **Use `csv.reader` instead of `csv.DictReader`.** Reading by column index avoids dictionary construction per row. Since we only need the "gloss" column, we can read the header once to find the index, then iterate with `csv.reader`:

   ```python
   reader = csv.reader(f)
   header = next(reader)
   gloss_col = header.index("gloss")
   glosses = sorted({row[gloss_col].strip().lower() for row in reader if row[gloss_col].strip()})
   ```

   This eliminates 250K `dict` constructions and the associated `dict.get` calls. Estimated savings: ~40–50% of function time.

2. **Cache the result.** The label map is deterministic for a given CSV. If called multiple times (e.g., during training setup), caching with `functools.lru_cache` or a module-level variable avoids re-reading the file.

3. **Use `pandas.read_csv`.** For very large CSVs (100K+ rows), pandas' C-accelerated CSV parser is 5–10× faster than Python's `csv` module.

---

## How to Reproduce

```bash
cd ml/
python -m profiling.profile_functions
```

This generates:
- **stdout**: Per-function cProfile tables + summary
- **`ml/profiling/results/*.prof`**: Binary profile files for `pstats` / `snakeviz`
- **`ml/profiling/results/profile_summary.json`**: Machine-readable summary

To explore a specific profile interactively:

```bash
python -c "
import pstats
p = pstats.Stats('ml/profiling/results/preprocess_video.prof')
p.sort_stats('cumulative')
p.print_stats(30)
"
```

Or with [`snakeviz`](https://jiffyclub.github.io/snakeviz/) for visual flame graphs:

```bash
pip install snakeviz
snakeviz ml/profiling/results/predict.prof
```

---

## Key Takeaways

1. **3D convolutions dominate inference and training** — `torch.conv3d` accounts for 60–96% of time in model-related functions. This is inherent to the I3D/R3D architecture. The most impactful optimizations are `torch.compile`, ONNX export, and mixed-precision inference.

2. **Backpropagation is 2× the forward pass** — in `i3d_train_one_epoch`, `run_backward` takes 66% of time vs. 30% for forward conv3d. Mixed-precision training (AMP) and freezing early backbone layers during fine-tuning can cut this significantly.

3. **Video preprocessing is resize-bound** — `cv2.resize` at 77% of `preprocess_video` time. Merging the two-step resize into a single operation would halve resize calls.

4. **CSV parsing overhead is avoidable** — `DictReader` creates unnecessary dictionaries. Switching to `csv.reader` with column-index access eliminates the overhead.

5. **Post-processing is negligible** — metric computation (accuracy, F1, confusion matrix) in `evaluate_model` takes <1% of total time. Optimization effort should focus exclusively on the model forward pass.
