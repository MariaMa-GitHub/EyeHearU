variable "name_prefix" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "security_group_id" {
  type = string
}

variable "batch_role_arn" {
  type = string
}

variable "pipeline_image_uri" {
  type = string
}

variable "s3_bucket_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "max_vcpus" {
  type    = number
  default = 4
}
