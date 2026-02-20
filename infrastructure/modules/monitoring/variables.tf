variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "alert_email" {
  type = string
}

variable "log_retention" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}
