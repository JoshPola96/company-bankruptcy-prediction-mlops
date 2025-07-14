# company-bankrupt-prediction/terraform/main.tf

# Configure the AWS provider
provider "aws" {
  region = var.aws_region # Set your desired AWS region from variables.tf
}

# --- VPC and Networking ---
# A simple VPC, public subnet, Internet Gateway, and Route Table for demonstration.
# For production, consider a more robust VPC design with private subnets, NAT Gateways, etc.
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = {
    Name = "mlops-bankrupt-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a" # Use a dynamic AZ based on region
  map_public_ip_on_launch = true
  tags = {
    Name = "mlops-bankrupt-public-subnet"
  }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
  tags = {
    Name = "mlops-bankrupt-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
  tags = {
    Name = "mlops-bankrupt-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# --- Security Group for Streamlit App (Port 8501) and SSH ---
resource "aws_security_group" "streamlit_sg" {
  vpc_id      = aws_vpc.main.id
  name        = "streamlit-app-sg"
  description = "Allow HTTP traffic on 8501 (Streamlit) and SSH (22)"

  # Ingress rule for Streamlit app
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # WARNING: For demo, allowing from all IPs. Restrict in production!
  }

  # Ingress rule for SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"] # WARNING: For demo, allowing from all IPs. Restrict in production!
  }

  # Egress rule (allow all outbound traffic)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "mlops-bankrupt-streamlit-sg"
  }
}

# --- S3 Bucket for Monitoring Data and Evidently Reports ---
resource "aws_s3_bucket" "monitoring_data" {
  bucket = "company-bankruptcy-prediction-monitoring"

  force_destroy = false

  tags = {
    Name = "company-bankruptcy-prediction-monitoring"
  }
}

# --- S3 Bucket for Monitoring Data and Evidently Reports ---
resource "aws_s3_bucket_policy" "monitoring_data_public" {
  bucket = aws_s3_bucket.monitoring_data.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "PublicReadGetObject"
        Effect = "Allow"
        Principal = "*"
        Action = "s3:GetObject"
        Resource = "${aws_s3_bucket.monitoring_data.arn}/*"
      }
    ]
  })
}

# Separate bucket ACL resource (newer Terraform AWS provider requirement)
resource "aws_s3_bucket_acl" "monitoring_data_acl" {
  depends_on = [aws_s3_bucket_ownership_controls.monitoring_data_acl_ownership]
  bucket     = aws_s3_bucket.monitoring_data.id
  acl        = "private"
}

resource "aws_s3_bucket_ownership_controls" "monitoring_data_acl_ownership" {
  bucket = aws_s3_bucket.monitoring_data.id
  rule {
    object_ownership = "BucketOwnerPreferred"
  }
}

# Block public access to the S3 bucket
resource "aws_s3_bucket_public_access_block" "monitoring_data_block" {
  bucket = aws_s3_bucket.monitoring_data.id

  block_public_acls       = true
  block_public_policy     = false
  ignore_public_acls      = true
  restrict_public_buckets = false
}

# --- IAM Role and Policy for EC2 to access S3 and ECR ---
resource "aws_iam_role" "ec2_s3_ecr_role" {
  name = "mlops-bankrupt-ec2-s3-ecr-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = "sts:AssumeRole",
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
      },
    ],
  })
  tags = {
    Name = "mlops-bankrupt-ec2-s3-ecr-role"
  }
}

resource "aws_iam_policy" "ec2_s3_ecr_policy" {
  name        = "mlops-bankrupt-ec2-s3-ecr-policy"
  description = "IAM policy for EC2 to read/write to S3 monitoring bucket and pull from ECR"

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ],
        Effect   = "Allow",
        Resource = [
          aws_s3_bucket.monitoring_data.arn,
          "${aws_s3_bucket.monitoring_data.arn}/*",
        ],
      },
      {
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetAuthorizationToken",
          "ecr:DescribeRepositories",
          "ecr:DescribeImages",
          "ecr:ListImages"
        ],
        Effect   = "Allow",
        Resource = "*",
      },
    ],
  })
}

resource "aws_iam_role_policy_attachment" "ec2_s3_ecr_attach" {
  role       = aws_iam_role.ec2_s3_ecr_role.name
  policy_arn = aws_iam_policy.ec2_s3_ecr_policy.arn
}

resource "aws_iam_instance_profile" "ec2_s3_ecr_profile" {
  name = "mlops-bankrupt-ec2-s3-ecr-profile"
  role = aws_iam_role.ec2_s3_ecr_role.name
}

# Data source to get AWS account ID for ECR URI
data "aws_caller_identity" "current" {}

# --- ECR Instance for Streamlit App Dovker Image ---
resource "aws_ecr_repository" "app_repo" {
  name = var.ecr_repo_name

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "company-bankrupt-prediction-app"
  }
}

# Add lifecycle policy to manage image retention
resource "aws_ecr_lifecycle_policy" "app_repo_policy" {
  repository = aws_ecr_repository.app_repo.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# --- EC2 Instance for Streamlit App ---
resource "aws_instance" "mlops_server" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.streamlit_sg.id]
  associate_public_ip_address = true
  key_name                    = var.key_name
  iam_instance_profile        = aws_iam_instance_profile.ec2_s3_ecr_profile.name

    # User data script
user_data = base64encode(<<EOF
#!/bin/bash
exec > /var/log/user-data.log 2>&1

echo "=== Updating system packages ==="
apt-get update -y

echo "=== Installing dependencies ==="
apt-get install -y ca-certificates curl gnupg lsb-release unzip

echo "=== Installing AWS CLI v2 ==="
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install --update
rm -rf awscliv2.zip aws/
ln -sf /usr/local/bin/aws /usr/bin/aws

echo "=== Installing Docker (robust) ==="
apt-get update -y
apt-get install -y apt-transport-https ca-certificates curl software-properties-common lsb-release

# Add Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

echo "=== Docker installed successfully ==="

echo "=== Waiting for IAM role and Docker setup ==="
sleep 30

echo "=== Testing AWS credentials and ECR availability ==="
for i in {1..10}; do
  if aws sts get-caller-identity --region ${var.aws_region} && \
     aws ecr describe-repositories --repository-names ${var.ecr_repo_name} --region ${var.aws_region}; then
    echo "AWS credentials and ECR repository ready"
    break
  else
    echo "Waiting for AWS credentials/ECR repository, attempt \$i/10, sleeping 30 seconds..."
    sleep 30
  fi
done

echo "=== Logging in to ECR ==="
for i in {1..5}; do
  if aws ecr get-login-password --region ${var.aws_region} | \
     docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com; then
    echo "ECR login successful"
    break
  else
    echo "ECR login failed (attempt \$i/5), retrying in 20 seconds..."
    sleep 20
  fi
done

echo "=== Pulling Docker image ==="
image_pulled=false
for i in {1..5}; do
  if docker pull ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repo_name}:${var.docker_image_tag}; then
    echo "Docker image pulled successfully"
    image_pulled=true
    break
  else
    echo "Docker pull failed (attempt \$i/5), retrying in 20 seconds..."
    sleep 20
  fi
done

if [ "\$image_pulled" = false ]; then
  echo "ERROR: Failed to pull Docker image after multiple attempts."
  exit 1
fi

echo "=== Creating systemd service file ==="
cat <<EOT > /etc/systemd/system/streamlit-app.service
[Unit]
Description=Streamlit App Docker Container
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
User=root
ExecStartPre=-/usr/bin/docker stop streamlit-app-production
ExecStartPre=-/usr/bin/docker rm streamlit-app-production
ExecStart=/usr/bin/docker run --rm -p 8501:8501 --name streamlit-app-production -e AWS_DEFAULT_REGION=${var.aws_region} ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repo_name}:${var.docker_image_tag}
ExecStop=/usr/bin/docker stop streamlit-app-production

[Install]
WantedBy=multi-user.target
EOT

echo "=== Enabling and starting systemd service ==="
systemctl daemon-reload
systemctl enable streamlit-app.service
systemctl start streamlit-app.service

echo "=== Finished. Check 'sudo systemctl status streamlit-app.service' and 'docker ps' to verify. ==="
EOF
)

  tags = {
    Name = "company-bankrupt-prediction-server"
  }
}

# --- Outputs ---

output "ecr_repository_url" {
  value       = aws_ecr_repository.app_repo.repository_url
  description = "URL of the ECR repository"
}

output "ecr_repository_name" {
  value       = aws_ecr_repository.app_repo.name
  description = "Name of the ECR repository"
}

output "instance_public_ip" {
  value       = aws_instance.mlops_server.public_ip
  description = "Public IP address of the Streamlit application server."
}

output "streamlit_app_url" {
  value       = "http://${aws_instance.mlops_server.public_ip}:8501"
  description = "URL to access the Streamlit application."
}

output "bucket_name" {
  value       = aws_s3_bucket.monitoring_data.id
  description = "Name of the S3 bucket for monitoring data and reports."
}

output "ssh_command" {
  value       = "ssh -i ${var.key_name}.pem ubuntu@${aws_instance.mlops_server.public_ip}"
  description = "SSH command to connect to the instance."
}
