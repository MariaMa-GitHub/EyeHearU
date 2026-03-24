# EyeHearU Migration and Training Playbook (New AWS Account + Modal + SFT)

This document summarizes what we completed and provides a reproducible, command-first tutorial.

Goals:

- Migrate data from the old AWS account to the new account
- Make Terraform work in the new account and point to the new data bucket
- Run training on Modal (smoke first, then full training)
- Run SFT from an existing checkpoint (warm start)

---

## 0. What Was Completed

Completed:

- New account data bucket: `eye-hear-u-public-data-ca1`
- Core training data migrated (`processed/mvp/`)
- Terraform `apply` succeeded in the new account
- Terraform outputs now point to the new bucket
- Modal training pipeline ran successfully and produced checkpoints
- Code was pushed to branch:
  - `chloe/modal-i3d-sft-aws-migration`

---

## 1. Cross-Account Data Migration (S3)

### 1.1 Prepare Two AWS Profiles

- Old account profile (example): `eyehearu`
- New account profile (example): `public`

Verify:

```bash
aws sts get-caller-identity --profile eyehearu --region ca-central-1
aws sts get-caller-identity --profile public --region ca-central-1
```

### 1.2 Create Target Bucket in New Account

```bash
aws s3 mb s3://eye-hear-u-public-data-ca1 --region ca-central-1 --profile public
```

### 1.3 Migrate Core Training Data (MVP)

```bash
mkdir -p /tmp/eyehearu-migrate
aws s3 sync s3://eye-hear-u-dev-data/processed/mvp/ /tmp/eyehearu-migrate/processed/mvp/ --profile eyehearu --region ca-central-1
aws s3 sync /tmp/eyehearu-migrate/processed/mvp/ s3://eye-hear-u-public-data-ca1/processed/mvp/ --profile public --region ca-central-1
```

### 1.4 Validate Migration

```bash
aws s3 ls s3://eye-hear-u-dev-data/processed/mvp/ --recursive --summarize --profile eyehearu --region ca-central-1
aws s3 ls s3://eye-hear-u-public-data-ca1/processed/mvp/ --recursive --summarize --profile public --region ca-central-1
```

---

## 2. Switch Terraform to New Account

### 2.1 Code Changes Already Made

Implemented changes:

- `infrastructure/variables.tf`
  - Added `existing_data_bucket_name`
- `infrastructure/main.tf`
  - Supports reusing an existing bucket (no forced new data lake creation)
- `infrastructure/outputs.tf`
  - Outputs now consistently return the active bucket in use
- `infrastructure/environments/dev.tfvars`
  - `existing_data_bucket_name = "eye-hear-u-public-data-ca1"`

### 2.2 Use a Dedicated State Bucket in New Account (Critical)

Avoid continuing to use the old account state bucket:

```bash
aws s3 mb s3://eye-hear-u-terraform-state-public --region ca-central-1 --profile public
aws s3api put-bucket-versioning --bucket eye-hear-u-terraform-state-public --versioning-configuration Status=Enabled --profile public --region ca-central-1
aws s3api put-bucket-encryption --bucket eye-hear-u-terraform-state-public --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' --profile public --region ca-central-1
```

### 2.3 Initialize and Deploy

```bash
cd infrastructure

AWS_PROFILE=public terraform init -reconfigure \
  -backend-config="bucket=eye-hear-u-terraform-state-public" \
  -backend-config="key=infrastructure/public-dev.tfstate" \
  -backend-config="region=ca-central-1"

AWS_PROFILE=public terraform plan -var-file=environments/dev.tfvars
AWS_PROFILE=public terraform apply -var-file=environments/dev.tfvars
```

Success signal:

- Output contains `s3_bucket_name = eye-hear-u-public-data-ca1`

---

## 3. Dataset Changes in This Repo

Yes, dataset-related changes are already in this repo, mainly in:

- `docs/i3d_s3_repro_guide.md`
- `data/scripts/prepare_i3d_from_s3.py`
- `data/scripts/plan_i3d_splits.py`
- `ml/i3d_msft/dataset.py`
- `ml/i3d_msft/s3_data.py`
- `ml/i3d_msft/train.py`

What changed (high level):

- Added an S3-to-I3D split export flow so training CSVs match Microsoft I3D expectations.
- Added robust split planning with versioned plans and rollback through `ACTIVE_PLAN.json`.
- Implemented signer-disjoint ASL Citizen eval split generation and kept supplemental data train-only.
- Added optional filtering for missing/unreadable clips before training.
- Added S3-aware data loading in the training stack (plan resolution, split download, clip sync).

MS-ASL vocabulary expansion update (2026-03-24):

- Added expansion list: `data/scripts/mvp_glosses_msasl_expand_v1.txt` (39 candidate glosses).
- Added utility script: `data/scripts/boost_low_glosses_to_s3.py` for low-gloss augmentation from MS-ASL + ASL Citizen with deduplication.
- Downloaded and processed additional MS-ASL data for expansion glosses.
- Uploaded expansion metadata:
  - `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v1.csv`
- Updated processed metadata:
  - `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/processed_clips.csv`
  - `s3://eye-hear-u-public-data-ca1/processed/mvp/processed_clips.csv`
- Net result from expansion processing: `+59` processed clips across `27` new glosses.

Additional expansion rounds (v5-v11, with bad-fragment filtering + full-video fallback):

- Round v5:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v5.txt`
  - download: `69` success / `104` failed
  - processed: `+152` clips, `50` new glosses, `21` bad/too-short skipped
  - metadata total: `1879 -> 2031`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v5.csv`
- Round v6:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v6.txt`
  - download: `66` success / `99` failed
  - processed: `+120` clips, `48` new glosses, `15` bad/too-short skipped
  - metadata total: `2031 -> 2151`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v6.csv`
- Round v7:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v7.txt`
  - download: `80` success / `105` failed
  - processed: `+125` clips, `50` new glosses, `14` bad/too-short skipped
  - metadata total: `2151 -> 2276`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v7.csv`
- Round v8:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v8.txt`
  - download: `68` success / `73` failed
  - processed: `+124` clips, `49` new glosses, `13` bad/too-short skipped
  - metadata total: `2276 -> 2400`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v8.csv`
- Round v9:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v9.txt`
  - download: `68` success / `80` failed
  - processed: `+103` clips, `47` new glosses, `14` bad/too-short skipped
  - metadata total: `2400 -> 2503`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v9.csv`
- Round v10:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v10.txt`
  - download: `62` success / `73` failed
  - processed: `+100` clips, `44` new glosses, `15` bad/too-short skipped
  - metadata total: `2503 -> 2603`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v10.csv`
- Round v11:
  - glossary: `data/scripts/mvp_glosses_msasl_expand_v11.txt`
  - download: `50` success / `71` failed
  - processed: `+91` clips, `45` new glosses, `2` bad/too-short skipped
  - metadata total: `2603 -> 2694`
  - artifacts: `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/ingested_msasl_expanded_v11.csv`
- Shared updated metadata keys after each round:
  - `s3://eye-hear-u-public-data-ca1/processed/mvp/metadata/processed_clips.csv`
  - `s3://eye-hear-u-public-data-ca1/processed/mvp/processed_clips.csv`

How rollback works:

- Each plan is stored under:
  - `s3://<bucket>/processed/mvp/i3d/split_plans/<plan_id>/`
- Active plan pointer:
  - `s3://<bucket>/processed/mvp/i3d/split_plans/ACTIVE_PLAN.json`
- To rollback, just repoint `ACTIVE_PLAN.json` to a previous `plan_id`.

Quick commands:

```bash
# Create a new plan candidate
PIPELINE_ENV=dev AWS_REGION=ca-central-1 python data/scripts/plan_i3d_splits.py \
  --mvp \
  --plan-id candidate-ac-eval-v3 \
  --drop-missing-s3 \
  --sample-s3-check 120

# Activate the candidate
PIPELINE_ENV=dev AWS_REGION=ca-central-1 python data/scripts/plan_i3d_splits.py \
  --mvp \
  --activate-plan candidate-ac-eval-v3

# Roll back to old plan
PIPELINE_ENV=dev AWS_REGION=ca-central-1 python data/scripts/plan_i3d_splits.py \
  --mvp \
  --activate-plan candidate-ac-eval-v2
```

---

## 4. Data Sources

Primary and supplemental data sources used in this workflow:

- **ASL Citizen** (primary dataset):
  - Main source for train/val/test evaluation protocol.
  - Signer-disjoint evaluation splits are preserved/enforced for clean generalization testing.
- **WLASL** (supplemental):
  - Used as additional training data when label overlap and filtering rules allow.
  - Not used as the evaluation benchmark in this setup.
- **MS-ASL** (supplemental):
  - Used as additional training data only.
  - Kept out of val/test to avoid cross-dataset evaluation leakage.

In short:

- Train can include ASL Citizen + supplemental datasets.
- Validation/test are ASL Citizen-focused and signer-disjoint for fair evaluation.

Data location conventions in S3:

- Processed dataset root: `s3://<bucket>/processed/mvp/`
- I3D split plans: `s3://<bucket>/processed/mvp/i3d/split_plans/<plan_id>/`
- Active split pointer: `s3://<bucket>/processed/mvp/i3d/split_plans/ACTIVE_PLAN.json`

---

## 5. Why We Train on Modal

Infrastructure in the new AWS account is in place, but GPU quota and operational friction can slow iteration.

Recommended setup:

- Platform layer: AWS (`S3`/`ECR`/`ECS`/`Batch`/`Terraform`)
- Training layer: Modal (faster startup and more stable iteration loop)

---

## 6. Modal Training Setup

### 4.1 Install and Authenticate

```bash
pip install modal
modal setup
```

### 4.2 Configure AWS Secret (for S3 access in Modal containers)

```bash
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=<new-account-ak> \
  AWS_SECRET_ACCESS_KEY=<new-account-sk> \
  AWS_REGION=ca-central-1
```

---

## 7. Training Commands (Smoke / Full / SFT)

Entry script:

- `modal_train_i3d.py`

### 5.1 Smoke Run (validate the full path first)

```bash
cd /Users/chloe/EyeHearU
modal run modal_train_i3d.py \
  --bucket eye-hear-u-public-data-ca1 \
  --plan-id candidate-ac-eval-v2 \
  --epochs 1 \
  --clip-limit 200
```

### 5.2 Full Training

```bash
modal run modal_train_i3d.py \
  --bucket eye-hear-u-public-data-ca1 \
  --plan-id candidate-ac-eval-v2 \
  --epochs 20 \
  --batch-size 6 \
  --num-workers 2
```

### 5.3 SFT (continue from existing weights)

`ml/i3d_msft/train.py` supports:

- `--init-checkpoint-s3-key`
- `--init-strict` (optional)

SFT command:

```bash
modal run modal_train_i3d.py \
  --bucket eye-hear-u-public-data-ca1 \
  --plan-id candidate-ac-eval-v2 \
  --epochs 20 \
  --batch-size 6 \
  --num-workers 2 \
  --init-checkpoint-s3-key models/i3d/modal/candidate-ac-eval-v2/<run_id>/best_model.pt
```

---

## 8. Output Locations

Artifacts are uploaded to:

- `s3://eye-hear-u-public-data-ca1/models/i3d/modal/<plan_id>/<run_id>/best_model.pt`
- `s3://eye-hear-u-public-data-ca1/models/i3d/modal/<plan_id>/<run_id>/run_metadata.json`

---

## 9. Common Troubleshooting

### 7.1 `SignatureDoesNotMatch`

Usually caused by mismatched AK/SK for the new account. Recreate the access key and rerun `aws configure --profile public`.

### 7.2 Terraform `AccessDenied` (CreateRepository / CreateCluster / CreateLogGroup)

The new account user lacks permissions. Ask your admin to grant required service permissions (or temporary `AdministratorAccess`).

### 7.3 Terraform backend `403`

You are still pointing to the old account state bucket. Use the new account state bucket and rerun `terraform init -reconfigure`.

### 7.4 `modal_train_i3d.py` not found

You are running from the wrong directory. Run from repo root, or reference the script with `../modal_train_i3d.py`.

---

## 10. Cost Notes

- Terraform-created resources (such as NAT/ALB) continue to incur costs.
- Modal GPU jobs are billed by runtime.
- After training, recommended cleanup:
  - Destroy unused cloud resources with `terraform destroy` (or disable by module)
  - Keep only required S3 data and checkpoints

