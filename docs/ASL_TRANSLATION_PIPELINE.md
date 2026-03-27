# ASL → English pipeline (single clip and multi-clip)

This document describes the **end-to-end translation path** implemented in Eye Hear U: what is wired together, what the outputs mean, and how that relates to “accurate translation.”

## What the product actually does

| Layer | Single clip (`/predict`, app **Single sign**) | Multi-clip (`/predict/sentence`, app **Multi-sign**) |
|--------|-----------------------------------------------|------------------------------------------------------|
| **Input** | One short video | Ordered list of videos (one isolated sign per clip, max 12) |
| **Vision** | I3D → top‑k **gloss** labels (English-like lemmas from training) | Same model, **batched** per clip (`predict_batch`) |
| **Sequence** | N/A | **Beam search** over per-clip top‑k hypotheses |
| **Language model** | N/A | **Gloss n-gram LM** (`GlossBeamLM`: trigram with bigram backoff) loaded from `backend/data/gloss_lm.json` |
| **“English” string** | Top‑1 `PredictionResponse.sign` (and TTS) | `SentencePredictionResponse.english`: **join** of `best_glosses` + **light polish** (see below) |

The mobile app lets the user choose **Single sign** vs **Multi-sign** on the camera screen; that only changes which API is called after capture (`predictSign` vs accumulating clips and `predictSentence`).

### What `english` is (multi-clip response)

`gloss_sequence_to_english` in `backend/app/services/gloss_to_english.py`:

- Does **not** run machine translation or a large language model.
- Takes the beam‑chosen ordered gloss list, joins with spaces, applies surface rules (e.g. `_` → space, lone `i` → `I`, capitalizes first character, adds `.` if missing).

So the result is a **single line of gloss lemmas**, formatted for reading aloud — **not** a guarantee of grammatically perfect or idiomatic English (no automatic tense, articles, or reordering).

### “Accurate translation” vs this stack

- **Classifier accuracy** depends on video quality, signer variation, and the **856-class** gloss model (see README / evaluation docs).
- **Sequence quality** (multi-clip) additionally depends on **per-clip** errors, **beam / LM** weights, and how well `gloss_lm.json` reflects real multi-sign statistics (see rebuilding LM below).
- **Fluent English prose** is **out of scope** for the current backend; improving that would require a separate **gloss-to-English** model or rules beyond join + polish.

The pipeline is **implemented, tested in CI (backend + mobile)**, and **error-handled** (validation, 4xx/5xx, empty inputs). It does **not** by itself satisfy a strict reading of “always accurate ASL→English translation” if that means **human-quality sentences**.

## End-to-end data flow (multi-clip)

```
Mobile (Multi-sign)
  → record / pick video per gloss, ordered URIs
  → POST /api/v1/predict/sentence  (multipart field `files` repeated, order preserved)

FastAPI predict_sentence
  → preprocess each clip → tensor list
  → predict_batch(model, tensors) → List[List[{sign, confidence}]]  # top-k per clip
  → beam_search(candidates, gloss_lm, beam_size, lm_weight)
  → SentencePredictionResponse: clips, beam[], best_glosses, english
```

Single-clip path skips beam and LM:

```
POST /api/v1/predict (field `file`)
  → preprocess → predict → top-1 sign + top_k
```

## Key source files

| Component | Location |
|-----------|----------|
| Single-sign route | `backend/app/routers/predict.py` → `predict_sign` |
| Multi-sign route | `backend/app/routers/predict.py` → `predict_sentence` |
| Batched inference | `backend/app/services/model_service.py` → `predict_batch` |
| Beam search | `backend/app/services/beam_search.py` |
| Gloss LM load | `backend/app/services/gloss_lm.py` → `load_gloss_lm`, `GlossBeamLM` |
| Gloss line formatting | `backend/app/services/gloss_to_english.py` |
| LM JSON builder (offline) | `backend/app/services/lm_builder.py`, `backend/scripts/build_gloss_lm.py` |
| LM + startup wiring | `backend/app/main.py` (lifespan), `backend/app/config.py` (`gloss_lm_path`) |
| Mobile: mode + APIs | `mobile/app/camera.tsx`, `mobile/services/api.ts` (`predictSign`, `predictSentence`) |

## Rebuilding `gloss_lm.json`

From the repo root (after `cd backend`):

```bash
PYTHONPATH=. python scripts/build_gloss_lm.py \
  --label-map ../ml/i3d_label_map_mvp-sft-full-v1.json \
  --out data/gloss_lm.json
```

Optional **richer** n-grams from your own ordered gloss sentences (one sentence per line, whitespace-separated glosses matching the label map):

```bash
PYTHONPATH=. python scripts/build_gloss_lm.py \
  --label-map ../ml/i3d_label_map_mvp-sft-full-v1.json \
  --sequences path/to/gloss_lines.txt \
  --out data/gloss_lm.json
```

Redeploy or restart the API after replacing the file.

## Related docs

- [User guide](USER_GUIDE.md) — how to use Single vs Multi-sign in the app  
- [Developer guide](DEVELOPER_GUIDE.md) — run backend/mobile locally  
- [Testing](TESTING.md) — pytest / Jest coverage for this pipeline  
- [Preprocessing](PREPROCESSING.md) — video → tensor (must match training)
