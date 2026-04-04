# ASL → English pipeline (single clip and multi-clip)

This document describes the **end-to-end translation path** implemented in Eye Hear U: what is wired together, what the outputs mean, and how that relates to “accurate translation.”

**Related:** [Preprocessing](PREPROCESSING.md) (video → tensor), [Developer guide](DEVELOPER_GUIDE.md) (endpoints and code map).

## What the product actually does

| Layer | Single clip (`/predict`, app **Single sign**) | Multi-clip (`/predict/sentence`, app **Multi-sign**) |
|--------|-----------------------------------------------|------------------------------------------------------|
| **Input** | One short video | Ordered list of videos (one isolated sign per clip, max 12) |
| **Vision** | I3D → top‑k **gloss** labels (English-like lemmas from training) | Same model, **batched** per clip (`predict_batch`) |
| **Sequence** | N/A | **Beam search** over per-clip top‑k hypotheses |
| **Language model** | N/A | **Gloss n-gram LM** (`GlossBeamLM`: trigram with bigram backoff) loaded from `backend/data/gloss_lm.json` |
| **“English” string** | Top‑1 `PredictionResponse.sign` (and TTS) | See **English modes** below |

The mobile app lets the user choose **Single sign** vs **Multi-sign** on the camera screen; that only changes which API is called after capture (`predictSign` vs accumulating clips and `predictSentence`).

### English modes (`GLOSS_ENGLISH_MODE` in `.env` / `Settings`)

| Mode | Best hypothesis (`english`, `best_glosses`) | Each `beam[]` row |
|------|---------------------------------------------|-------------------|
| **`rule`** (default) | `gloss_sequence_to_english`: join glosses + light polish (no LLM) | Same rule formatter |
| **`t5`** | Local **FLAN-T5** rewrite of the beam-chosen gloss list | **Rule** only (avoids running T5 once per beam hypothesis on CPU) |
| **`bedrock`** | **Amazon Bedrock** rewrite when the API succeeds; **rule** fallback on error (logged at WARNING) | **Rule** only (avoids many paid calls per request) |

So for **`t5`** and **`bedrock`**, the **display line** (`english`) can read more like a sentence, while **alternate beam rows** stay as gloss-style lines unless you change this policy.

### What `english` is in **`rule`** mode

`gloss_sequence_to_english` in `backend/app/services/gloss_to_english.py`:

- Does **not** run machine translation or a large language model.
- Takes the beam‑chosen ordered gloss list, joins with spaces, applies surface rules (e.g. `_` → space, lone `i` → `I`, capitalizes first character, adds `.` if missing).

The result is a **single line of gloss lemmas**, formatted for reading aloud — **not** a guarantee of grammatically perfect or idiomatic English.

### Beam search guardrails

If **any** clip has an **empty** top‑k list (no classifier candidates), `beam_search` raises `ValueError` and `POST /predict/sentence` responds with **400** and a clear message. That avoids returning an empty `best_glosses` / `english` with no explanation.

### “Accurate translation” vs this stack

- **Classifier accuracy** depends on video quality, signer variation, and the **856-class** gloss model (see README / evaluation docs).
- **Sequence quality** (multi-clip) additionally depends on **per-clip** errors, **beam / LM** weights, and how well `gloss_lm.json` reflects real multi-sign statistics (see rebuilding LM below).
- **Fluent, idiomatic English** is **not guaranteed**: in **`rule`** mode the line is **gloss-oriented** formatting only; **`t5`** / **`bedrock`** can produce **more natural prose** for the **best** path when enabled, subject to model quality, latency, and fallbacks.

The pipeline is **implemented, tested in CI (backend + ML + mobile)**, and **error-handled** (validation, 4xx/5xx, empty inputs). It does **not** by itself satisfy a strict reading of “always accurate ASL→English translation” if that means **human-quality sentences** in all modes.

## End-to-end data flow (multi-clip)

```
Mobile (Multi-sign)
  → record / pick video per gloss, ordered URIs
  → POST /api/v1/predict/sentence  (multipart field `files` repeated, order preserved)

FastAPI predict_sentence
  → preprocess each clip → tensor list
  → predict_batch(model, tensors) → List[List[{sign, confidence}]]  # top-k per clip
  → beam_search(candidates, gloss_lm, beam_size, lm_weight)  # fails fast if any clip has empty top-k
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
| Optional rewriters | `gloss_to_english_t5.py`, `gloss_to_english_bedrock.py` |
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
