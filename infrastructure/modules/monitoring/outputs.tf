output "log_group_name" {
  value = aws_cloudwatch_log_group.pipeline.name
}

output "log_group_arn" {
  value = aws_cloudwatch_log_group.pipeline.arn
}

output "sns_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
