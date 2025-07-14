import os

import pendulum
from airflow.operators.bash import BashOperator
from dotenv import load_dotenv

from airflow import DAG

# Load env vars from .env
load_dotenv()

ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
ECR_REPO_NAME = os.environ.get("ECR_REPO_NAME")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION")
REPO_URL = f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}"

with DAG(
    dag_id="streamlit_deployment_pipeline_dag",
    schedule=None,
    start_date=pendulum.datetime(2025, 7, 12, tz="UTC"),
    catchup=False,
    tags=["mlops", "ci-cd", "streamlit"],
) as dag:
    # Step 1: Create ECR repo only (targeted)
    create_ecr_repo = BashOperator(
        task_id="create_ecr_repo",
        bash_command="""
            echo "Creating ECR repository via Terraform target..."
            cd /opt/airflow/terraform/
            terraform init
            terraform apply -target=aws_ecr_repository.app_repo -auto-approve
        """,
    )

    # Step 2: Prepare Docker build context
    prepare_context = BashOperator(
        task_id="prepare_build_context",
        bash_command="""
            echo "Preparing streamlit_app build context..."
            rm -rf /opt/airflow/streamlit_app/src
            rm -rf /opt/airflow/streamlit_app/data
            cp -r /opt/airflow/src /opt/airflow/streamlit_app/
            mkdir -p /opt/airflow/streamlit_app/data
            cp -r /opt/airflow/data/models /opt/airflow/streamlit_app/data/
            echo "Copied src and data/models into streamlit_app."
            ls -la /opt/airflow/streamlit_app/
        """,
    )

    # Step 3: Build Docker image
    build_image = BashOperator(
        task_id="build_docker_image",
        bash_command="""
            echo "Building Docker image..."
            echo "Repository: {{ params.repo_name }}"
            echo "Tag: {{ data_interval_end.strftime('%Y%m%d%H%M%S') }}"
            docker build -f /opt/airflow/streamlit_app/Dockerfile \
                -t {{ params.repo_name }}:{{ data_interval_end.strftime('%Y%m%d%H%M%S') }} \
                /opt/airflow/streamlit_app
        """,
        params={"repo_name": ECR_REPO_NAME},
        do_xcom_push=False,
    )

    # Step 4: Cleanup build context
    cleanup_context = BashOperator(
        task_id="cleanup_build_context",
        bash_command="""
            echo "Cleaning up streamlit_app build context..."
            rm -rf /opt/airflow/streamlit_app/src
            rm -rf /opt/airflow/streamlit_app/data
            echo "Build context cleaned up."
        """,
    )

    # Step 5: Push image to ECR
    push_image = BashOperator(
        task_id="push_to_ecr",
        bash_command="""
        echo "=== ECR Push Process Started ==="
        aws sts get-caller-identity || { echo "AWS credentials not available"; exit 1; }

        # Define a consistent timestamp variable
        TIMESTAMP_TAG="{{ data_interval_end.strftime('%Y%m%d%H%M%S') }}"
        LATEST_TAG="latest"

        # Check if image exists locally
        if ! docker images {{ params.repo_name }}:$TIMESTAMP_TAG | grep -q {{ params.repo_name }}; then
            echo "Docker image with tag $TIMESTAMP_TAG not found locally"
            echo "Available images:"
            docker images {{ params.repo_name }}
            exit 1
        fi

        echo "Logging into ECR..."
        for i in {1..3}; do
            if aws ecr get-login-password --region {{ params.region }} | \
                docker login --username AWS --password-stdin {{ params.account_id }}.dkr.ecr.{{ params.region }}.amazonaws.com; then
                echo "ECR login successful"
                break
            else
                echo "ECR login attempt $i failed, retrying..."
                sleep 5
            fi
        done

        echo "Tagging image for ECR push..."
        docker tag {{ params.repo_name }}:$TIMESTAMP_TAG \
            {{ params.repo_url }}:$TIMESTAMP_TAG
        docker tag {{ params.repo_name }}:$TIMESTAMP_TAG \
            {{ params.repo_url }}:$LATEST_TAG

        echo "Pushing images with tags $TIMESTAMP_TAG and 'latest'..."
        docker push {{ params.repo_url }}:$TIMESTAMP_TAG
        docker push {{ params.repo_url }}:$LATEST_TAG

        echo "Verifying images were pushed successfully..."
        aws ecr describe-images --repository-name {{ params.repo_name }} --region {{ params.region }}

        echo "=== ECR Push Process Completed ==="
    """,
        params={
            "repo_name": ECR_REPO_NAME,
            "repo_url": REPO_URL,
            "account_id": ACCOUNT_ID,
            "region": AWS_REGION,
        },
    )

    wait_for_ecr = BashOperator(
        task_id="wait_for_ecr_propagation",
        bash_command="""
        echo "Waiting for ECR image to be available..."
        sleep 30
        echo "Verifying image exists in ECR..."
        aws ecr describe-images --repository-name {{ params.repo_name }} --region {{ params.region }} --image-ids imageTag={{ data_interval_end.strftime('%Y%m%d%H%M%S') }}
    """,
        params={
            "repo_name": ECR_REPO_NAME,
            "region": AWS_REGION,
        },
    )

    # Step 6: Deploy full infra (EC2, VPC, SG, etc.)
    deploy_infra = BashOperator(
        task_id="deploy_with_terraform",
        bash_command="""
        echo "Deploying infrastructure with Terraform..."
        cd /opt/airflow/terraform/
        terraform init
        terraform refresh
        terraform apply -auto-approve
    """,
    )

    # Dependencies
    (
        create_ecr_repo
        >> prepare_context
        >> build_image
        >> cleanup_context
        >> push_image
        >> wait_for_ecr
        >> deploy_infra
    )
