output "pipeline_repository_url" {
  value = aws_ecr_repository.pipeline.repository_url
}

output "api_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "repository_arn" {
  value = aws_ecr_repository.pipeline.arn
}
