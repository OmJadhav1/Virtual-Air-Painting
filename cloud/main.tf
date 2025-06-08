provider "aws" {
  region = "ap-south-1"
}

# S3 Bucket for logs or intermediate storage
resource "aws_s3_bucket" "bucket" {
  bucket = "hand-gesture-recognition-bucket"
}

# ECS Cluster
resource "aws_ecs_cluster" "gesture_cluster" {
  name = "gesture-recognition-cluster"
}

# ECR Repository
resource "aws_ecr_repository" "gesture_repo" {
  name = "hand-gesture-recognition-repo"
}

# IAM Role for ECS Task Execution
resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecsTaskExecutionRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Attach Policies to ECS Role
resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Task Definition
resource "aws_ecs_task_definition" "gesture_task" {
  family                   = "gesture-recognition-task"
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_execution_role.arn
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"

  container_definitions = jsonencode([
    {
      name      = "gesture-container",
      image     = "${aws_ecr_repository.gesture_repo.repository_url}:latest",
      memory    = 512,
      cpu       = 256,
      essential = true,
      portMappings = [
        {
          containerPort = 5000,
          hostPort      = 5000,
          protocol      = "tcp"
        }
      ]
    }
  ])
}


# ECS Service
resource "aws_ecs_service" "gesture_service" {
  name            = "gesture-recognition-service"
  cluster         = aws_ecs_cluster.gesture_cluster.id
  task_definition = aws_ecs_task_definition.gesture_task.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = ["subnet-0a1b2c3d4e5f67890"] 
    security_groups  = [aws_security_group.gesture_sg.id]
    assign_public_ip = true
  }
}

# Application Load Balancer
resource "aws_lb" "gesture_lb" {
  name               = "gesture-recognition-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.gesture_sg.id]
  subnets            = ["subnet-0a1b2c3d4e5f67890"]
}

# Load Balancer Listener
resource "aws_lb_listener" "gesture_lb_listener" {
  load_balancer_arn = aws_lb.gesture_lb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"
    fixed_response {
      status_code  = "200"
      message_body = "OK"
      content_type = "text/plain"
    }
  }
}

# ------------------------------
# Amazon Kinesis Data Stream
# ------------------------------
resource "aws_kinesis_stream" "gesture_stream" {
  name             = "gesture-frame-stream"
  shard_count      = 1
  retention_period = 24
}

# ------------------------------
# EC2 Instance (Optional Consumer/Processor)
# ------------------------------
resource "aws_key_pair" "gesture_key" {
  key_name   = "gesture-key"
  public_key = file("~/.ssh/id_rsa.pub") # Replace with your public key
}

resource "aws_instance" "gesture_ec2" {
  ami           = "ami-0c5204531f799e0c6" # Amazon Linux 2, adjust as needed
  instance_type = "t2.micro"
  key_name      = aws_key_pair.gesture_key.key_name
  subnet_id     = "subnet-0a1b2c3d4e5f67890"
  vpc_security_group_ids = [aws_security_group.gesture_sg.id]

  tags = {
    Name = "Gesture-Kinesis-Consumer"
  }

  user_data = <<EOF
#!/bin/bash
yum update -y
yum install -y python3 git
pip3 install boto3
EOF
}
