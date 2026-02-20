variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "alert_email" {
  description = "Email address for SNS alert notifications"
  type        = string
}

# ── Batch sizing ─────────────────────────────────────────────────
variable "batch_max_vcpus" {
  description = "Maximum vCPUs for the AWS Batch compute environment"
  type        = number
  default     = 4
}

# ── ECS sizing ───────────────────────────────────────────────────
variable "ecs_task_cpu" {
  description = "CPU units for the ECS Fargate task (256 = 0.25 vCPU)"
  type        = number
  default     = 256
}

variable "ecs_task_memory" {
  description = "Memory (MB) for the ECS Fargate task"
  type        = number
  default     = 512
}

# ── CloudWatch ───────────────────────────────────────────────────
variable "cloudwatch_log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30
}
