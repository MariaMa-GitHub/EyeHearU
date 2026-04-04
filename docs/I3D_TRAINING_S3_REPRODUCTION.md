# I3D training reproduction (S3)

**See also:** [Documentation index](README.md) · [Modal / AWS SFT migration](MODAL_AWS_SFT_MIGRATION.md)

This guide documents an end-to-end, reproducible workflow for:

- building I3D-compatible split files from S3 data,
- training on AWS machines (not local-only),
- switching between split plans safely,
- rolling back instantly if results degrade.

The flow is designed around the current bucket:

- `s3://eye-hear-u-dev-data/`

---

## 1) What Is Backed Up

All split variants are stored in the same S3 bucket:

- Previous default split export:
  - `s3://eye-hear-u-dev-data/processed/mvp/i3d/splits/{train,val,test}.csv`
- Versioned candidate plans:
  - `s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/candidate-ac-eval-v1/...`
  - `s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/candidate-ac-eval-v4/...`
- Active pointer:
  - `s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/ACTIVE_PLAN.json`

Current active plan:

- `candidate-ac-eval-v4`

---

## 2) Prerequisites

On the machine running scripts:

- Python 3.10+
- AWS CLI configured and authenticated
- `PIPELINE_ENV=dev`
- AWS region: `ca-central-1`

Quick check:

```bash
aws sts get-caller-identity
aws s3 ls s3://eye-hear-u-dev-data
```

---

## 3) Build MVP Dataset on S3 (Data Construction)

Run from repo root:

```bash
export PIPELINE_ENV=dev
export AWS_REGION=ca-central-1
```

Install pipeline deps (once):

```bash
pip install -r data/scripts/requirements.txt
```

Build dataset artifacts to S3:

```bash
python data/scripts/ingest_asl_citizen.py --mvp
python data/scripts/extract_mvp_videos_from_zip.py --skip-existing
python data/scripts/preprocess_clips.py --source asl_citizen --mvp
python data/scripts/build_unified_dataset.py --mvp
python data/scripts/validate.py --mvp
```

Expected key outputs:

- `s3://eye-hear-u-dev-data/processed/mvp/processed_clips.csv`
- `s3://eye-hear-u-dev-data/processed/mvp/label_map.json`
- `s3://eye-hear-u-dev-data/processed/mvp/dataset_stats.json`
- clips under `s3://eye-hear-u-dev-data/processed/mvp/clips/`

---

## 4) Build/Refresh Versioned Split Plans

Use this script to generate a signer-disjoint ASL-Citizen eval split and keep supplemental data train-only:

- Script: `data/scripts/plan_i3d_splits.py`

Example (create a new candidate, keep ACTIVE unchanged):

```bash
PIPELINE_ENV=dev AWS_REGION=ca-central-1 \
python data/scripts/plan_i3d_splits.py \
  --mvp \
  --plan-id candidate-ac-eval-v3 \
  --drop-missing-s3 \
  --sample-s3-check 120
```

What it writes:

- `.../split_plans/<plan_id>/splits/train.csv`
- `.../split_plans/<plan_id>/splits/val.csv`
- `.../split_plans/<plan_id>/splits/test.csv`
- `.../split_plans/<plan_id>/manifest.json`

---

## 5) Activate a Plan (and Roll Back)

Activate chosen plan:

```bash
PIPELINE_ENV=dev AWS_REGION=ca-central-1 \
python data/scripts/plan_i3d_splits.py \
  --mvp \
  --activate-plan candidate-ac-eval-v4
```

Verify active pointer:

```bash
aws s3 cp s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/ACTIVE_PLAN.json -
```

Rollback is the same command with another plan ID:

```bash
PIPELINE_ENV=dev AWS_REGION=ca-central-1 \
python data/scripts/plan_i3d_splits.py \
  --mvp \
  --activate-plan candidate-ac-eval-v1
```

---

## 6) Prepare AWS Training Inputs

For Microsoft I3D loader (`user,filename,gloss`), use:

- video root from processed clips:
  - `s3://eye-hear-u-dev-data/processed/mvp/clips/`
- split CSVs from active/candidate plan:
  - `s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/<plan_id>/splits/`

On the AWS training instance:

```bash
mkdir -p /opt/eyehearu/i3d_data/{clips,splits}

aws s3 sync s3://eye-hear-u-dev-data/processed/mvp/clips/ /opt/eyehearu/i3d_data/clips/
aws s3 cp s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/candidate-ac-eval-v4/splits/train.csv /opt/eyehearu/i3d_data/splits/train.csv
aws s3 cp s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/candidate-ac-eval-v4/splits/val.csv /opt/eyehearu/i3d_data/splits/val.csv
aws s3 cp s3://eye-hear-u-dev-data/processed/mvp/i3d/split_plans/candidate-ac-eval-v4/splits/test.csv /opt/eyehearu/i3d_data/splits/test.csv
```

Note: Microsoft `aslcitizen_dataset.py` builds paths as `datadir + filename`.
If your CSV filenames are like `train/gloss/clip.mp4`, set `datadir` to:

- `/opt/eyehearu/i3d_data/clips/`

---

## 7) Run Training (Microsoft I3D Repo)

In Microsoft I3D training script, set:

- `video_base_path = '/opt/eyehearu/i3d_data/clips/'`
- `train_file = '/opt/eyehearu/i3d_data/splits/train.csv'`
- `test_file = '/opt/eyehearu/i3d_data/splits/val.csv'`  (validation file in their naming)

Then run:

```bash
python3 aslcitizen_training.py
```

For final evaluation:

- use `/opt/eyehearu/i3d_data/splits/test.csv`

---

## 8) Run Training (Vendored in This Repo)

We also copied the required Microsoft I3D components into this repo under:

- `ml/i3d_msft/pytorch_i3d.py`
- `ml/i3d_msft/videotransforms.py`
- `ml/i3d_msft/dataset.py`
- `ml/i3d_msft/train.py` (S3-adapted trainer)

Run from `ml/`:

```bash
pip install -r requirements.txt
python -m i3d_msft.train \
  --bucket eye-hear-u-dev-data \
  --region ca-central-1 \
  --plan-id candidate-ac-eval-v4 \
  --epochs 20
```

Notes:

- If `--plan-id` is omitted, it uses `ACTIVE_PLAN.json`.
- It auto-downloads required train/val clips from:
  - `s3://eye-hear-u-dev-data/processed/mvp/clips/`
- It saves checkpoints under:
  - `ml/workdir/i3d_msft/checkpoints/<plan_id>/`

Smoke test (recommended first, avoids long stalls on bad clips):

```bash
OPENCV_LOG_LEVEL=ERROR python -m i3d_msft.train \
  --bucket eye-hear-u-dev-data \
  --region ca-central-1 \
  --plan-id candidate-ac-eval-v4 \
  --epochs 1 \
  --batch-size 2 \
  --num-workers 0 \
  --clip-limit 80 \
  --device cpu
```

Notes for stability:

- By default, trainer filters split rows to files that exist locally and are decodable.
- `--clip-limit` is applied with val coverage so smoke runs still keep validation samples.
- Use `--no-verify-readable` only when your dataset has already been cleaned.

---

## 9) Recommended A/B Evaluation Protocol

To compare two plans safely:

1. Keep ACTIVE unchanged.
2. Train plan A and plan B with identical hyperparameters and seeds.
3. Compare validation metrics and stability across seeds.
4. Activate winner via `--activate-plan`.
5. Keep losing plan for rollback/history.

Store outputs in separate S3 prefixes, for example:

- `s3://eye-hear-u-dev-data/models/i3d/plan_candidate-ac-eval-v1/...`
- `s3://eye-hear-u-dev-data/models/i3d/plan_candidate-ac-eval-v4/...`

---

## 10) Commands Summary

Create candidate:

```bash
PIPELINE_ENV=dev AWS_REGION=ca-central-1 python data/scripts/plan_i3d_splits.py --mvp --plan-id candidate-ac-eval-vX --drop-missing-s3 --sample-s3-check 120
```

Activate candidate:

```bash
PIPELINE_ENV=dev AWS_REGION=ca-central-1 python data/scripts/plan_i3d_splits.py --mvp --activate-plan candidate-ac-eval-vX
```

Rollback:

```bash
PIPELINE_ENV=dev AWS_REGION=ca-central-1 python data/scripts/plan_i3d_splits.py --mvp --activate-plan candidate-ac-eval-v1
```

