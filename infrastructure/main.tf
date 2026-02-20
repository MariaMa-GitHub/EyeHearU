terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state in S3 (bootstrap this bucket manually or via a separate config)
  backend "s3" {
    bucket         = "eye-hear-u-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "eye-hear-u-terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "eye-hear-u"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

locals {
  name_prefix = "eye-hear-u-${var.environment}"
}

# ── Networking ───────────────────────────────────────────────────
module "networking" {
  source      = "./modules/networking"
  name_prefix = local.name_prefix
  aws_region  = var.aws_region
}

# ── IAM Roles ────────────────────────────────────────────────────
module "iam" {
  source         = "./modules/iam"
  name_prefix    = local.name_prefix
  s3_bucket_arn  = module.s3.bucket_arn
  ecr_repo_arn   = module.ecr.repository_arn
  log_group_arn  = module.monitoring.log_group_arn
}

# ── S3 Data Lake ─────────────────────────────────────────────────
module "s3" {
  source      = "./modules/s3"
  name_prefix = local.name_prefix
  environment = var.environment
}

# ── ECR Container Registry ──────────────────────────────────────
module "ecr" {
  source      = "./modules/ecr"
  name_prefix = local.name_prefix
}

# ── AWS Batch (Pipeline Jobs) ────────────────────────────────────
module "batch" {
  source              = "./modules/batch"
  name_prefix         = local.name_prefix
  subnet_ids          = module.networking.private_subnet_ids
  security_group_id   = module.networking.batch_security_group_id
  batch_role_arn      = module.iam.batch_execution_role_arn
  pipeline_image_uri  = module.ecr.pipeline_repository_url
  s3_bucket_name      = module.s3.bucket_name
  environment         = var.environment
  max_vcpus           = var.batch_max_vcpus
}

# ── ECS Fargate (Inference API) ──────────────────────────────────
module "ecs" {
  source             = "./modules/ecs"
  name_prefix        = local.name_prefix
  vpc_id             = module.networking.vpc_id
  public_subnet_ids  = module.networking.public_subnet_ids
  private_subnet_ids = module.networking.private_subnet_ids
  ecs_role_arn       = module.iam.ecs_execution_role_arn
  ecs_task_role_arn  = module.iam.ecs_task_role_arn
  api_image_uri      = module.ecr.api_repository_url
  s3_bucket_name     = module.s3.bucket_name
  task_cpu           = var.ecs_task_cpu
  task_memory        = var.ecs_task_memory
  environment        = var.environment
}

# ── Monitoring & Alerts ──────────────────────────────────────────
module "monitoring" {
  source         = "./modules/monitoring"
  name_prefix    = local.name_prefix
  environment    = var.environment
  alert_email    = var.alert_email
  log_retention  = var.cloudwatch_log_retention_days
}
