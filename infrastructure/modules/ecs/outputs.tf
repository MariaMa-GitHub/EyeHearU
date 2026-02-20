output "cluster_id" {
  value = aws_ecs_cluster.api.id
}

output "service_name" {
  value = aws_ecs_service.api.name
}

output "alb_dns_name" {
  value = aws_lb.api.dns_name
}
