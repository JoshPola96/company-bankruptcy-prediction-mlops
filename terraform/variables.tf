# company-bankrupt-prediction/terraform/variables.tf
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "ap-south-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
}

variable "key_name" {
  description = "SSH key name to access EC2"
  type        = string
}

variable "monitoring_s3_bucket_name" {
  description = "Globally unique S3 bucket name for monitoring"
  type        = string
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository"
  type        = string
  default     = "company-bankrupt-prediction-streamlit-app"
}

variable "docker_image_tag" {
  description = "Tag for the Docker image"
  type        = string
  default     = "latest"
}