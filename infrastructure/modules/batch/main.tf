resource "aws_batch_compute_environment" "pipeline" {
  compute_environment_name = "${var.name_prefix}-pipeline-compute"
  type                     = "MANAGED"
  service_role             = var.batch_role_arn

  compute_resources {
    type      = "FARGATE"
    max_vcpus = var.max_vcpus

    subnets            = var.subnet_ids
    security_group_ids = [var.security_group_id]
  }
}

resource "aws_batch_job_queue" "pipeline" {
  name     = "${var.name_prefix}-pipeline-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.pipeline.arn
  }
}

resource "aws_batch_job_definition" "ingest" {
  name = "${var.name_prefix}-ingest"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image   = "${var.pipeline_image_uri}:latest"
    command = ["scripts/ingest_asl_citizen.py"]

    resourceRequirements = [
      { type = "VCPU", value = "1" },
      { type = "MEMORY", value = "2048" },
    ]

    environment = [
      { name = "PIPELINE_ENV", value = var.environment },
      { name = "S3_BUCKET", value = var.s3_bucket_name },
    ]

    executionRoleArn = var.batch_role_arn
    jobRoleArn       = var.batch_role_arn
  })
}

resource "aws_batch_job_definition" "preprocess" {
  name = "${var.name_prefix}-preprocess"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image   = "${var.pipeline_image_uri}:latest"
    command = ["scripts/preprocess_clips.py"]

    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "4096" },
    ]

    environment = [
      { name = "PIPELINE_ENV", value = var.environment },
      { name = "S3_BUCKET", value = var.s3_bucket_name },
    ]

    executionRoleArn = var.batch_role_arn
    jobRoleArn       = var.batch_role_arn
  })
}

resource "aws_batch_job_definition" "build_dataset" {
  name = "${var.name_prefix}-build-dataset"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image   = "${var.pipeline_image_uri}:latest"
    command = ["scripts/build_unified_dataset.py"]

    resourceRequirements = [
      { type = "VCPU", value = "1" },
      { type = "MEMORY", value = "2048" },
    ]

    environment = [
      { name = "PIPELINE_ENV", value = var.environment },
      { name = "S3_BUCKET", value = var.s3_bucket_name },
    ]

    executionRoleArn = var.batch_role_arn
    jobRoleArn       = var.batch_role_arn
  })
}

resource "aws_batch_job_definition" "validate" {
  name = "${var.name_prefix}-validate"
  type = "container"

  platform_capabilities = ["FARGATE"]

  container_properties = jsonencode({
    image   = "${var.pipeline_image_uri}:latest"
    command = ["scripts/validate.py"]

    resourceRequirements = [
      { type = "VCPU", value = "1" },
      { type = "MEMORY", value = "2048" },
    ]

    environment = [
      { name = "PIPELINE_ENV", value = var.environment },
      { name = "S3_BUCKET", value = var.s3_bucket_name },
    ]

    executionRoleArn = var.batch_role_arn
    jobRoleArn       = var.batch_role_arn
  })
}
