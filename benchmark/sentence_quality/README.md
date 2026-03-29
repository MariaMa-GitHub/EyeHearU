# Sentence quality benchmark (rule vs T5 vs Bedrock) and beam-search ablation

This benchmark evaluates how well each gloss-to-English mode converts the **same gloss sequence** into natural English.

## What to compare

- **rule**: beam search + LM + join/polish
- **t5**: beam search + FLAN-T5 rewrite
- **bedrock**: beam search + Bedrock (Claude Haiku) rewrite

## Metrics (automatic)

For each system against your reference sentence:

- `token_precision`, `token_recall`, `token_f1` (class-style precision/recall/F1 on words)
- `sentence_bleu` (1–4 gram BLEU with brevity penalty)
- `rougeL_recall` (sequence overlap recall)
- `exact_match` (strict normalized equality)
- `capitalization_ok`, `punctuation_ok` (surface quality)

## 1) Prepare data

Create CSV like `data_template.csv` with:

- `case_id`
- `glosses` (use `|` separator, e.g. `nurse|yes|table`)
- `reference` (human target English sentence)

Use 30–100 cases for a meaningful comparison.

## 2) Generate model outputs

From the **`backend`** directory (repo root as `PYTHONPATH` parent as usual):

```bash
cd backend
export PYTHONPATH=..
python ../benchmark/sentence_quality/evaluate.py generate \
  --input ../benchmark/sentence_quality/data_template.csv \
  --output ../benchmark/sentence_quality/results/predictions.csv
```

This calls:

- `app.services.gloss_to_english.gloss_sequence_to_english`
- `app.services.gloss_to_english_t5.gloss_sequence_to_english_t5`
- `app.services.gloss_to_english_bedrock.gloss_sequence_to_english_bedrock`

Bedrock uses your backend `.env` settings (`BEDROCK_REGION`, `BEDROCK_MODEL_ID`).

## 3) Score outputs

```bash
cd backend
export PYTHONPATH=..
python ../benchmark/sentence_quality/evaluate.py score \
  --predictions ../benchmark/sentence_quality/results/predictions.csv \
  --out ../benchmark/sentence_quality/results/metrics.json
```

## 4) Suggested reporting categories

Use three sections in your report:

1. **Faithfulness**: token recall/F1, ROUGE-L (did it keep meaning)
2. **Fluency**: punctuation/capitalization + manual 1–5 rating
3. **Overall quality**: sentence BLEU + side-by-side examples

Add 5–10 qualitative examples where systems differ the most.

## Beam vs no-beam ablation

To test whether beam search over the per-clip top-k matrix helps, use `evaluate_ablation.py`.

### Input format

Use JSON like `data_ablation_template.json`:

- `case_id`
- `reference`
- `candidates`: list of clips; each clip is a list of top-k candidates `{"sign": "...", "confidence": ...}`.

This matches the matrix view: **rows** = clip positions, **columns** = top-k hypotheses for that clip.

### Why greedy matched beam on the production LM

The repo’s `backend/data/gloss_lm.json` is mostly `<s> → gloss` mass; **after the first gloss**, bigrams are nearly Laplace-uniform. Then total score ≈ sum of per-clip log-confidence, and **beam ≈ greedy**.

To **demonstrate** beam changing the path, this folder includes `gloss_lm_ablation.json` (tiny LM with strong bigrams) and near-tied fake confidences in `data_ablation_template.json`. The script uses that LM by default (`--lm-json` optional).

### Generate ablation predictions

```bash
cd backend
export PYTHONPATH=..
python ../benchmark/sentence_quality/evaluate_ablation.py generate \
  --input ../benchmark/sentence_quality/data_ablation_template.json \
  --output ../benchmark/sentence_quality/results/ablation_predictions.json \
  --beam-size 8 \
  --lm-weight 2.0
```

Use `--lm-json path/to/gloss_lm.json` to score against the **production** LM instead (expect little or no greedy/beam gap).

### Score ablation

```bash
cd backend
export PYTHONPATH=..
python ../benchmark/sentence_quality/evaluate_ablation.py score \
  --predictions ../benchmark/sentence_quality/results/ablation_predictions.json \
  --out ../benchmark/sentence_quality/results/ablation_metrics.json
```

### Compared systems

- `greedy_rule`, `greedy_t5`, `greedy_bedrock`
- `beam_rule`, `beam_t5`, `beam_bedrock`

You can isolate:

1. **Decoding**: greedy vs beam (same rewriter)
2. **Rewriter**: rule vs T5 vs Bedrock (same decoding)

### Example aggregate numbers (one project run; your mileage will vary)

**Main benchmark (gloss → natural sentence)**

| Mode    | F1    | BLEU  | ROUGE-L |
|---------|-------|-------|---------|
| rule    | 0.819 | 0.060 | 0.683   |
| t5      | 0.732 | 0.093 | 0.696   |
| bedrock | 0.839 | 0.248 | 0.786   |

**Ablation (beam vs greedy)**

| System         | F1    | BLEU  | ROUGE-L |
|----------------|-------|-------|---------|
| greedy_rule    | 0.317 | 0.000 | 0.307   |
| greedy_t5      | 0.277 | 0.000 | 0.307   |
| greedy_bedrock | 0.284 | 0.000 | 0.333   |
| beam_rule      | 0.643 | 0.000 | 0.553   |
| beam_t5        | 0.552 | 0.000 | 0.553   |
| beam_bedrock   | 0.728 | 0.156 | 0.660   |

Treat these as **illustrative**; re-run on your data and environment.
