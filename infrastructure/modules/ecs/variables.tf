variable "name_prefix" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_ids" {
  type = list(string)
}

variable "private_subnet_ids" {
  type = list(string)
}

variable "ecs_role_arn" {
  type = string
}

variable "ecs_task_role_arn" {
  type = string
}

variable "api_image_uri" {
  type = string
}

variable "s3_bucket_name" {
  type = string
}

variable "task_cpu" {
  type    = number
  default = 256
}

variable "task_memory" {
  type    = number
  default = 512
}

variable "environment" {
  type = string
}
