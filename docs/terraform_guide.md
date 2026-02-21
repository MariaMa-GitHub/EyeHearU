# Terraform Infrastructure Guide — Eye Hear U

This guide explains how the Terraform infrastructure works, how to deploy it, and how to use it for the disaster recovery demo (Part 5). Every team member should read this before touching AWS.

---

## 1. What Terraform Does

Terraform is a tool that lets you describe cloud infrastructure in `.tf` files (declarative configuration), then creates, updates, or destroys real AWS resources to match what the code says. Instead of clicking around the AWS Console, you write code, run a command, and Terraform figures out what needs to be created, modified, or deleted.

The core lifecycle is four commands:

```bash
terraform init      # downloads provider plugins (AWS), sets up state backend
terraform plan      # shows what WOULD change (dry run — creates nothing)
terraform apply     # actually creates/updates resources on AWS
terraform destroy   # tears everything down (used for Part 5: disaster recovery)
```

`plan` is always safe to run — it changes nothing. Always run `plan` before `apply` to review what will happen.

---

## 2. How State Works

Terraform keeps a **state file** (`terraform.tfstate`) that records the mapping between your `.tf` code and the real AWS resources it created. This is how Terraform knows:

- "S3 bucket X already exists, no change needed"
- "ECS service Y was deleted manually, need to recreate it"
- "Variable Z changed from 256 to 512, need to update the task definition"

**Our state is stored remotely in S3** so the entire team shares it, and a DynamoDB table prevents two people from running `apply` simultaneously:

```hcl
backend "s3" {
  bucket         = "eye-hear-u-terraform-state"
  key            = "infrastructure/terraform.tfstate"
  region         = "ca-central-1"
  dynamodb_table = "eye-hear-u-terraform-locks"
  encrypt        = true
}
```

**Important:** Never manually edit the state file. If something goes wrong, use `terraform import` or `terraform state rm` to fix it.

---

## 3. Project Structure

```
infrastructure/
├── main.tf                  # Root module: provider config, wires all modules together
├── variables.tf             # Input variables (environment, region, sizing)
├── outputs.tf               # Values exported after apply (URLs, ARNs, bucket names)
│
├── environments/
│   ├── dev.tfvars           # Dev overrides (small instances, 7-day logs)
│   ├── staging.tfvars       # Staging overrides (medium, 30-day logs)
│   └── prod.tfvars          # Production overrides (full-size, 90-day logs)
│
└── modules/
    ├── s3/                  # Data lake bucket
    ├── ecr/                 # Docker image registries
    ├── batch/               # Pipeline compute (AWS Batch)
    ├── ecs/                 # Inference API (Fargate + ALB)
    ├── iam/                 # Roles and permissions
    ├── networking/          # VPC, subnets, NAT gateway
    └── monitoring/          # CloudWatch logs, SNS alerts
```

---

## 4. How Modules Work

Each module is a self-contained unit that manages one logical piece of infrastructure. Every module has three files:


| File           | Purpose                                                             |
| -------------- | ------------------------------------------------------------------- |
| `main.tf`      | The actual AWS resources (e.g., `aws_s3_bucket`, `aws_ecs_cluster`) |
| `variables.tf` | Inputs the module accepts (e.g., `name_prefix`, `environment`)      |
| `outputs.tf`   | Values the module exposes to other modules (e.g., `bucket_arn`)     |


### 4.1 What Each Module Creates

`**s3/**` — Data Lake Bucket

- One S3 bucket per environment (`eye-hear-u-{env}-data`)
- Versioning enabled (critical for disaster recovery — deleted objects can be restored)
- Server-side encryption (AES-256)
- Public access fully blocked
- Lifecycle rule: move `raw/` objects to Infrequent Access after 30 days (cost saving)

`**ecr/**` — Container Registries

- Two ECR repositories: `data-pipeline` (for pipeline scripts) and `backend-api` (for FastAPI)
- Image scanning on push (security)
- Lifecycle policy: keep only the last 10 images (cost saving)

`**networking/**` — VPC and Networking

- One VPC (`10.0.0.0/16`)
- 2 public subnets (for the ALB — internet-facing)
- 2 private subnets (for ECS tasks and Batch jobs — no direct internet access)
- Internet gateway (for public subnets)
- NAT gateway (so private subnets can reach the internet to pull Docker images and access S3)
- Security group for Batch jobs (egress-only)

`**iam/**` — Roles and Permissions

- **Batch execution role**: can read/write S3, pull images from ECR, write CloudWatch logs
- **ECS execution role**: can pull images from ECR (standard AWS managed policy)
- **ECS task role**: can read S3 (to load the model) and write CloudWatch metrics

Each role follows the principle of least privilege — only the permissions it needs.

`**batch/`** — Pipeline Compute

- One Fargate compute environment with configurable max vCPUs
- One job queue
- Four job definitions (one per pipeline stage: ingest, preprocess, build, validate)
- Each job runs the `data-pipeline` Docker image with the appropriate Python script as the command

`**ecs/`** — Inference API

- One ECS Fargate cluster
- Task definition for the FastAPI container (port 8000)
- ECS service with desired count = 1
- Application Load Balancer (ALB) with health check on `/health`
- Two security groups: ALB (accepts port 80 from the internet) and ECS tasks (accepts port 8000 from ALB only)

`**monitoring/`** — Logs and Alerts

- CloudWatch log group (`/eye-hear-u/{env}`) with configurable retention
- SNS topic for alerts with email subscription
- Pipeline failure alarm (fires when validation reports errors)
- API latency alarm (fires when average response time exceeds 5 seconds)

### 4.2 How Modules Connect

The root `main.tf` passes outputs from one module as inputs to another. For example, the IAM module needs the S3 bucket ARN to write its permission policies:

```hcl
module "iam" {
  source        = "./modules/iam"
  s3_bucket_arn = module.s3.bucket_arn          # ← output from S3 module
  ecr_repo_arn  = module.ecr.repository_arn     # ← output from ECR module
  log_group_arn = module.monitoring.log_group_arn
}
```

Terraform automatically resolves these dependencies and creates resources in the right order:

```
networking ──────────────────┐
                             │
monitoring ──────────────────┤
                             │
s3 ──────┐                   │
         ├──→ iam ──────┬────┤
ecr ─────┘              │    │
                        │    │
                        ├──→ batch  (needs subnets, security groups, IAM role, ECR image)
                        │
                        └──→ ecs    (needs subnets, IAM roles, ECR image, VPC)
```

You never need to worry about ordering — Terraform handles it.

---

## 5. How Environments Work

The three `.tfvars` files under `environments/` are variable overrides. They all use the exact same Terraform code, but with different values.

When you run Terraform, you pick which environment by passing the file:

```bash
terraform apply -var-file=environments/dev.tfvars      # deploy dev
terraform apply -var-file=environments/staging.tfvars   # deploy staging
terraform apply -var-file=environments/prod.tfvars      # deploy prod
```

### 5.1 What Differs Per Environment


| Property         | dev                   | staging                   | prod                   |
| ---------------- | --------------------- | ------------------------- | ---------------------- |
| S3 bucket        | `eye-hear-u-dev-data` | `eye-hear-u-staging-data` | `eye-hear-u-prod-data` |
| Batch max vCPUs  | 2                     | 4                         | 8                      |
| ECS task CPU     | 256 (0.25 vCPU)       | 512 (0.5 vCPU)            | 1024 (1 vCPU)          |
| ECS task memory  | 512 MB                | 1024 MB                   | 2048 MB                |
| Log retention    | 7 days                | 30 days                   | 90 days                |
| S3 force_destroy | true (easy teardown)  | true                      | **false** (safety net) |


### 5.2 Why Separate Environments

- **dev**: cheapest. Used for testing Terraform changes and pipeline code. Safe to destroy and recreate freely.
- **staging**: production-like config but with real data. Used to verify everything works before promoting to prod.
- **prod**: the real thing. Serves the actual inference API. S3 bucket cannot be force-destroyed (requires manual emptying first as a safety check).

All resources are completely isolated — dev S3 buckets, IAM roles, ECS clusters, etc. are entirely separate from prod. No risk of accidentally breaking production while experimenting in dev.

---

## 6. Step-by-Step Deployment

### 6.1 Prerequisites

Install these on your machine:

```bash
# macOS
brew install terraform
brew install awscli
brew install docker

# Verify
terraform --version    # should be >= 1.5
aws --version
docker --version
```

Configure AWS credentials:

```bash
aws configure
# Enter:
#   AWS Access Key ID:     <your key>
#   AWS Secret Access Key: <your secret>
#   Default region name:   ca-central-1
#   Default output format: json
```

### 6.2 Bootstrap the State Backend (one-time, one person does this)

Before Terraform can manage anything, it needs an S3 bucket and DynamoDB table to store its own state. These are created manually:

```bash
# Create the state bucket
aws s3 mb s3://eye-hear-u-terraform-state --region ca-central-1

# Enable versioning on it
aws s3api put-bucket-versioning \
  --bucket eye-hear-u-terraform-state \
  --versioning-configuration Status=Enabled

# Create the lock table(not used since We replaced dynamodb_table with use_lockfile = true in your main.tf, so Terraform now handles locking natively via S3 — no DynamoDB table needed. It's one fewer resource to manage and pay for.)
aws dynamodb create-table \
  --table-name eye-hear-u-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region ca-central-1
```

### 6.3 Initialize Terraform

```bash
cd infrastructure
terraform init
```

This downloads the AWS provider plugin and connects to the remote state backend. You only need to run this once (or after changing providers/backend config).

### 6.4 Deploy Dev Environment

```bash
# Always plan first
terraform plan -var-file environments/dev.tfvars

# Review the output. It will show something like:
#   Plan: 25 to add, 0 to change, 0 to destroy.

# If it looks correct, apply
terraform apply -var-file environments/dev.tfvars

# Type "yes" when prompted.
```

After apply completes, Terraform prints the outputs:

```
Outputs:

api_load_balancer_dns = "eye-hear-u-dev-api-alb-1638616272.ca-central-1.elb.amazonaws.com"
batch_job_queue_arn = "arn:aws:batch:ca-central-1:772548857721:job-queue/eye-hear-u-dev-pipeline-queue"
cloudwatch_log_group = "/eye-hear-u/dev"
ecr_api_repo_url = "772548857721.dkr.ecr.ca-central-1.amazonaws.com/eye-hear-u-dev-backend-api"
ecr_pipeline_repo_url = "772548857721.dkr.ecr.ca-central-1.amazonaws.com/eye-hear-u-dev-data-pipeline"
s3_bucket_arn = "arn:aws:s3:::eye-hear-u-dev-data"
s3_bucket_name = "eye-hear-u-dev-data"
sns_topic_arn = "arn:aws:sns:ca-central-1:772548857721:eye-hear-u-dev-alerts"
```

### 6.5 Build and Push Docker Images

```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ca-central-1

# Log in to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and push the data pipeline image
cd data
docker build -t eye-hear-u-pipeline .
docker tag eye-hear-u-pipeline:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/eye-hear-u-dev-data-pipeline:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/eye-hear-u-dev-data-pipeline:latest

# Build and push the backend API image
cd ../backend
docker build -t eye-hear-u-api .
docker tag eye-hear-u-api:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/eye-hear-u-dev-backend-api:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/eye-hear-u-dev-backend-api:latest
```

### 6.6 Upload Raw Data to S3

```bash
# Upload ASL Citizen videos
aws s3 sync data/raw/asl_citizen s3://eye-hear-u-dev-data/raw/asl_citizen/ --storage-class STANDARD

# Upload WLASL videos
aws s3 sync data/raw/wlasl s3://eye-hear-u-dev-data/raw/wlasl/

# Upload MS-ASL videos
aws s3 sync data/raw/msasl s3://eye-hear-u-dev-data/raw/msasl/
```

### 6.7 Run the Pipeline on AWS Batch

```bash
# Submit ingestion jobs (these run in parallel)
aws batch submit-job \
  --job-name ingest-asl-citizen \
  --job-queue eye-hear-u-dev-pipeline-queue \
  --job-definition eye-hear-u-dev-ingest \
  --container-overrides '{"command":["scripts/ingest_asl_citizen.py"]}'

aws batch submit-job \
  --job-name ingest-wlasl \
  --job-queue eye-hear-u-dev-pipeline-queue \
  --job-definition eye-hear-u-dev-ingest \
  --container-overrides '{"command":["scripts/ingest_wlasl.py"]}'

aws batch submit-job \
  --job-name ingest-msasl \
  --job-queue eye-hear-u-dev-pipeline-queue \
  --job-definition eye-hear-u-dev-ingest \
  --container-overrides '{"command":["scripts/ingest_msasl.py"]}'

# After ingestion completes, submit preprocessing
aws batch submit-job \
  --job-name preprocess \
  --job-queue eye-hear-u-dev-pipeline-queue \
  --job-definition eye-hear-u-dev-preprocess

# Then build the unified dataset
aws batch submit-job \
  --job-name build-dataset \
  --job-queue eye-hear-u-dev-pipeline-queue \
  --job-definition eye-hear-u-dev-build-dataset

# Finally validate
aws batch submit-job \
  --job-name validate \
  --job-queue eye-hear-u-dev-pipeline-queue \
  --job-definition eye-hear-u-dev-validate
```

Monitor jobs in the AWS Console under **AWS Batch → Jobs**, or via CLI:

```bash
aws batch describe-jobs --jobs <job-id>
```

### 6.8 Deploy to Staging / Production

Once dev works, promote to staging and then prod:

```bash
terraform plan -var-file=environments/staging.tfvars
terraform apply -var-file=environments/staging.tfvars

terraform plan -var-file=environments/prod.tfvars
terraform apply -var-file=environments/prod.tfvars
```

---

## 7. Disaster Recovery (Part 5)

Part 5 requires a screen recording of deleting the production environment and restoring it with IaC. Here is the exact procedure:

### 7.1 Deletion (Record Screen From Here)

```bash
cd infrastructure

# Show what currently exists
terraform output

# Destroy everything
terraform destroy -var-file=environments/prod.tfvars
# Type "yes" when prompted

# Verify everything is gone
aws s3 ls | grep eye-hear-u-prod          # should return nothing
aws ecs list-clusters | grep eye-hear-u   # should return nothing
aws batch describe-compute-environments   # should not show our environment
```

### 7.2 Restoration

```bash
# Recreate all infrastructure from code
terraform apply -var-file=environments/prod.tfvars
# Type "yes"

# Verify resources exist
terraform output

# Restore S3 data (from versioned objects or re-upload)
aws s3 sync data/raw/ s3://eye-hear-u-prod-data/raw/

# Rebuild and push Docker images
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/eye-hear-u-prod-data-pipeline:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/eye-hear-u-prod-backend-api:latest

# Re-run the data pipeline to reconstruct the processed layer
aws batch submit-job --job-name restore-pipeline \
  --job-queue eye-hear-u-prod-pipeline-queue \
  --job-definition eye-hear-u-prod-ingest

# Force a new deployment of the ECS service to pick up the image
aws ecs update-service \
  --cluster eye-hear-u-prod-api-cluster \
  --service eye-hear-u-prod-api-service \
  --force-new-deployment

# Verify the API is healthy
curl http://$(terraform output -raw api_load_balancer_dns)/health
```

### 7.3 What Gets Restored


| Component                 | How It's Restored                                         |
| ------------------------- | --------------------------------------------------------- |
| S3 bucket + config        | `terraform apply` recreates with versioning + encryption  |
| S3 data (raw videos)      | Re-upload from local or from versioned objects            |
| S3 data (processed clips) | Re-run the pipeline from raw data                         |
| ECR repositories          | `terraform apply` recreates; re-push Docker images        |
| VPC + subnets + NAT       | `terraform apply` recreates networking                    |
| IAM roles + policies      | `terraform apply` recreates with correct permissions      |
| AWS Batch compute + jobs  | `terraform apply` recreates compute env + job defs        |
| ECS cluster + API service | `terraform apply` recreates; force new deployment         |
| ALB + health checks       | `terraform apply` recreates; DNS resolves automatically   |
| CloudWatch logs           | `terraform apply` recreates log group (old logs are lost) |
| SNS alerts                | `terraform apply` recreates topic (re-confirm email sub)  |


---

## 8. Common Operations

### Viewing what Terraform manages

```bash
terraform state list                    # list all managed resources
terraform state show module.s3          # show details of a specific resource
terraform output                        # show all outputs (URLs, ARNs)
```

### Making changes

Edit a `.tf` file, then:

```bash
terraform plan -var-file=environments/dev.tfvars    # see what would change
terraform apply -var-file=environments/dev.tfvars   # apply the change
```

### Viewing logs

```bash
# Pipeline job logs
aws logs tail /eye-hear-u/dev --follow

# API logs
aws logs tail /ecs/eye-hear-u-dev-api --follow
```

### Checking costs

Dev is intentionally minimal. Approximate monthly costs:


| Resource          | dev         | prod        |
| ----------------- | ----------- | ----------- |
| S3 storage        | ~$2         | ~$20        |
| NAT gateway       | ~$32        | ~$32        |
| ALB               | ~$16        | ~$16        |
| ECS Fargate       | ~$5         | ~$15        |
| Batch (on-demand) | ~$0 idle    | ~$0 idle    |
| **Total**         | **~$55/mo** | **~$83/mo** |


The NAT gateway is the biggest fixed cost. For dev, you can destroy the infrastructure when not in use and recreate it before demos to save money.

---

## 9. Rules for the Team

1. **Never edit AWS resources manually in the Console.** If you change something by hand, Terraform state will be out of sync and the next `apply` may break things.
2. **Always run `plan` before `apply`.** Read the output. If it says "destroy" on something you don't expect, stop and investigate.
3. **Work in dev first.** Never test changes directly in prod.
4. **Don't commit `.tfvars` files with real secrets.** Our current `.tfvars` files only contain non-sensitive values (sizing, email). If you add secrets (API keys, passwords), use environment variables or AWS Secrets Manager instead.
5. **One person applies at a time.** The DynamoDB lock table prevents concurrent applies, but coordinate with the team to avoid confusion.

