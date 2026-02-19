output "s3_bucket_name" {
  description = "Name of the data lake S3 bucket"
  value       = module.s3.bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the data lake S3 bucket"
  value       = module.s3.bucket_arn
}

output "ecr_pipeline_repo_url" {
  description = "ECR repository URL for the data pipeline image"
  value       = module.ecr.pipeline_repository_url
}

output "ecr_api_repo_url" {
  description = "ECR repository URL for the backend API image"
  value       = module.ecr.api_repository_url
}

output "api_load_balancer_dns" {
  description = "DNS name of the ALB fronting the inference API"
  value       = module.ecs.alb_dns_name
}

output "batch_job_queue_arn" {
  description = "ARN of the AWS Batch job queue for pipeline jobs"
  value       = module.batch.job_queue_arn
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = module.monitoring.log_group_name
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts"
  value       = module.monitoring.sns_topic_arn
}
