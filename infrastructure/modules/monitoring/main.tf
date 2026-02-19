resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/eye-hear-u/${var.environment}"
  retention_in_days = var.log_retention

  tags = {
    Name = "${var.name_prefix}-logs"
  }
}

resource "aws_sns_topic" "alerts" {
  name = "${var.name_prefix}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── Pipeline Failure Alarm ───────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "pipeline_failure" {
  alarm_name          = "${var.name_prefix}-pipeline-failure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "PipelineValidationFailures"
  namespace           = "EyeHearU/${var.environment}"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Fires when the data pipeline validation step reports failures"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  treat_missing_data = "notBreaching"
}

# ── API High Latency Alarm ───────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "api_latency" {
  alarm_name          = "${var.name_prefix}-api-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 5
  alarm_description   = "API average response time exceeds 5 seconds"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  treat_missing_data = "notBreaching"
}
