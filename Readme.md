# 📈 End-to-End MLOps Pipeline for Company Bankruptcy Prediction

## 🌟 Project Overview

This project implements a robust, end-to-end MLOps pipeline designed to predict company bankruptcy. The problem of identifying companies at risk of bankruptcy is critical for investors, creditors, and business analysts to make informed decisions and mitigate financial risks. Early and accurate prediction allows stakeholders to intervene, adjust strategies, or protect investments before significant losses occur.

Our solution tackles this challenge by building an automated, self-healing machine learning system that not only trains and deploys a predictive model but also continuously monitors its performance and triggers retraining when necessary. This ensures that the model remains accurate and relevant in a constantly evolving financial landscape, solving the common MLOps challenges of model drift, manual deployment, and lack of reproducibility.

### 📊 Dataset Information

The datasets are sourced from three different providers and have been engineered and merged to create a comprehensive training dataset. The data exhibits **highly imbalanced characteristics** with significantly fewer samples of bankrupt companies compared to non-bankrupt ones. This imbalance has been mitigated as much as possible using SMOTE (Synthetic Minority Oversampling Technique) and other advanced training techniques, though some impact on model performance is still observable and expected given the nature of the problem domain.

## 🚀 MLOps Stack

This project leverages a modern MLOps stack to create a resilient and automated pipeline:

- **Workflow Orchestration:** **Apache Airflow** orchestrates the entire ML pipeline, managing data ingestion, preprocessing, model training, deployment, and continuous monitoring via a series of interconnected Directed Acyclic Graphs (DAGs), and is hosted and managed locally.
- **Infrastructure as Code (IaC):** **Terraform** provisions and manages all necessary cloud resources on AWS, ensuring consistent and repeatable infrastructure deployments.
- **Experiment Tracking & Model Registry:** **MLflow** is hosted and managed locally for logging experiment parameters, metrics, and artifacts. The best model is also promoted to production and is constantly updated with the best versions in each runs.
- **Containerization:** **Docker** is used to package both the Airflow environment and the Streamlit application for consistent, isolated deployments.
- **Model Serving:** **Streamlit** provides an intuitive, interactive web application to serve real-time bankruptcy predictions from the deployed model.
- **Monitoring & Data Quality:** **Evidently AI** is integrated into the pipeline to monitor for data drift in production. It compares live inference data against training data, triggering automated retraining when significant drift is detected.
- **Cloud Provider:** **Amazon Web Services (AWS)** hosts all the infrastructure, including EC2 for the Streamlit app, S3 for data storage, ECR for Docker images, and IAM for access management.

*Only the trained models and application infrastructure are deployed to AWS.*

## 📦 Project Structure

```
.
├── .github/                   # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── ci.yml
├── airflow/                   # Dockerized Airflow setup
│   ├── dags/                  # Airflow DAGs for pipeline orchestration
│   │   ├── data_download_dag.py
│   │   ├── data_consolidation_dag.py
│   │   ├── model_training_pipeline_dag.py
│   │   ├── streamlit_deployment_pipeline_dag.py
│   │   └── monitoring_pipeline_dag.py
│   ├── .env                   # Environment variables for Airflow Docker Compose (sensitive)
│   ├── .env_example.txt       # Template for Airflow-specific environment variables
│   ├── Dockerfile             # Dockerfile for Airflow custom image
│   ├── docker-compose.yaml    # Docker Compose configuration for local Airflow environment
│   ├── logs/                  # Airflow runtime logs (ignored by git)
│   ├── requirements-aws.txt   # AWS-specific dependencies for Airflow
│   ├── requirements-extra.txt # Extra Airflow dependencies
│   └── requirements-ml.txt    # ML-related dependencies for Airflow
├── data/                      # Placeholder for raw and processed data (populated by pipeline)
├── mlflow_setup/              # Local MLflow backend and artifact storage
│   ├── mlflow_artifacts/      # MLflow artifact storage (ignored by git)
│   └── mlruns/                # MLflow tracking runs (ignored by git)
├── notebooks/                 # Jupyter notebooks for EDA and experimentation
│   └── 01-eda.ipynb
├── src/                       # Python source code for pipeline stages and main app
│   ├── app.py                 # Streamlit web application main file
│   ├── __init__.py            # Python package initializer
│   ├── data_processing.py     # Data processing and transformation logic
│   ├── monitor_model.py       # Handles data drift monitoring logic
│   ├── train_model.py         # Handles model training and evaluation logic
│   └── unpause_all_dags.sh    # Script to unpause Airflow DAGs on startup
├── streamlit_app/             # Streamlit app Docker context
│   ├── Dockerfile
│   └── requirements.txt       # Streamlit app dependencies
├── terraform/                 # Terraform configuration files for AWS infrastructure
│   ├── .terraform/            # Terraform internal files (ignored by git)
│   ├── .terraform.lock.hcl
│   ├── main.tf
│   ├── terraform.tfstate      # Terraform state file (ignored by git)
│   ├── terraform.tfstate.backup
│   ├── terraform.tfvars       # Terraform variables (sensitive, ignored by git)
│   └── variables.tf
├── tests/                     # Unit and Integration Tests
│   ├── __init__.py            # Python package initializer
│   ├── test_unit_data.py
│   └── test_integration_mlflow.py
├── .env.example               # Project-wide environment variable template
├── .gitignore                 # Files and directories ignored by Git
├── .pre-commit-config.yaml    # Configuration for pre-commit hooks
├── Makefile                   # Automation for common development and CI tasks
├── pyproject.toml             # Project metadata and tool configurations
└── requirements.txt           # Main project Python dependencies
```

## 🛠️ Setup and Execution Guide

Follow these steps to set up and run the entire MLOps pipeline locally. The `Makefile` automates most of the complex commands.

### Prerequisites

Ensure you have the following installed on your machine:

- **Git:** For cloning the repository
- **Docker Desktop:** Includes Docker Engine and Docker Compose. [Download here](https://www.docker.com/products/docker-desktop/)
- **Terraform CLI:** [Download here](https://developer.hashicorp.com/terraform/downloads)
- **AWS CLI:** Configured with an AWS user that has programmatic access and sufficient permissions to create EC2 instances, S3 buckets, ECR repositories, and IAM roles. [Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/company-bankrupt-prediction.git
cd company-bankrupt-prediction
```

### Step 2: Configure Environment Variables

This project uses `.env` files for environment-specific configurations and secrets.

#### Airflow Environment Variables (`airflow/.env`)

Create `airflow/.env` from the example template:

```bash
cp airflow/.env_example.txt airflow/.env
```

Open `airflow/.env` and fill in your AWS credentials and configuration:

```env
AIRFLOW_UID=1000
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<YOUR_AWS_DEFAULT_REGION>
AWS_REGION=<YOUR_AWS_DEFAULT_REGION>
ECR_REPO_NAME=company-bankrupt-prediction-streamlit-app
AWS_ACCOUNT_ID=<YOUR_AWS_ACCOUNT_ID>
```

**Important Notes:**
- The `AIRFLOW_UID` should typically be set to `1000` for most systems
- The `docker-compose.yaml` in `airflow/` uses `airflow/.env` for all Airflow service containers
- Keep this file secure and never commit it to version control

#### Terraform Variables (`terraform/terraform.tfvars`)

Update the `terraform.tfvars` file with your specific AWS deployment configuration:

```hcl
aws_region                = "ap-south-1"
instance_type             = "t2.micro"
ami_id                    = "ami-02521d90e7410d9f0"
key_name                  = "mlops-zoomcamp"
monitoring_s3_bucket_name = "company-bankruptcy-prediction-monitoring"
```

### Step 3: Deploy AWS Infrastructure with Terraform

Navigate to the `terraform/` directory and provision your infrastructure:

```bash
cd terraform
terraform init
terraform apply --auto-approve
```

After successful execution, Terraform will output the public IP of your EC2 instance. Keep this handy for accessing your deployed Streamlit application.

### Step 4: Local Development Environment Setup

Set up a local Python virtual environment for linting and testing:

```bash
cd .. # Ensure you are in the project root
make setup
```

**Note:** The `make setup` command automatically creates a virtual environment, installs all required dependencies from `requirements.txt`, and prepares the environment for development. You must run this before proceeding with other Makefile commands.

To manually activate this environment for direct scripting:
```bash
source venv/bin/activate
```

### Step 5: Install Pre-commit Hooks

Enable automatic code quality checks before every commit:

```bash
make setup # Ensure local venv is set up
pre-commit install
```

Now, `git commit` will automatically format and lint your code based on `.pre-commit-config.yaml`.

### Step 6: Start All Local Services

Use the Makefile to start both Airflow and MLflow services:

```bash
make all_up
```

This command will:
- Format and lint your code
- Run all unit and integration tests
- Start the Dockerized Airflow services in detached mode
- Start the MLflow Tracking Server on your host machine

After completion, you can access:
- **Airflow UI:** `http://localhost:8080` (credentials: `airflow`/`airflow`)
- **MLflow UI:** `http://localhost:5000`

**Note:** The exact URLs will also be displayed in the terminal output from the Makefile commands for easy access.

### Step 7: Trigger the ML Pipeline

1. **Automatic DAG Unpausing:** The `airflow-scheduler` service automatically runs `src/unpause_all_dags.sh` on startup to unpause all DAGs
2. **Manual Execution:** From the Airflow UI (`http://localhost:8080`), you need to **manually trigger each DAG** in the following order:

   - `data_download_pipeline_dag`
   - `data_consolidation_pipeline_dag` 
   - `model_training_pipeline_dag` (⚠️ **May take up to 15 minutes**)
   - `streamlit_deployment_pipeline_dag`

**Important:** These DAGs are **not automatically chained** and must be run manually in sequence. Wait for each DAG to complete successfully before triggering the next one.

### Step 8: Access the Deployed Streamlit App

After the `streamlit_deployment_pipeline_dag` completes successfully:

1. **Wait for EC2 Setup:** Allow a few minutes for the EC2 instance to fully initialize
2. **Get Public IP:** Retrieve the **Public IPv4 address** from the AWS Management Console (EC2 Dashboard → Instances)
3. **Access Application:** Navigate to `http://<YOUR_EC2_PUBLIC_IP>:8501`

**Note:** The IP address and Evidently Bucket URL are also available in the Airflow DAG run logs.

### Step 9: Continuous Monitoring and Automated Retraining

The `monitoring_pipeline_dag` runs on a scheduled basis (configurable in the DAG file) and:

- Uses **Evidently AI** to analyze live inference data against training data for drift detection
- Requires a minimum of **30 inference logs** to perform meaningful drift analysis
- **Expected Behavior:** Due to the complexity and variance in financial data, drift detection will likely always trigger retraining when thresholds are exceeded. This demonstrates the robust self-healing mechanism of the pipeline
- **Automatic Pipeline Chaining:** When significant drift is detected, the monitoring DAG automatically triggers the **entire pipeline sequence** from data download through infrastructure deployment. This is the **only scenario** where the DAGs are automatically chained together.
- **Simulated Notifications:** The monitoring system creates a notification file to demonstrate alert capabilities when drift is detected
- Creates a powerful self-healing loop ensuring optimal model performance

## 🧹 Cleanup

### Stop Local Services

```bash
make all_down # Stops Airflow and MLflow services, removes Docker volumes
make clean    # Removes local build artifacts, venv, and generated data/MLflow files
```

### Destroy AWS Infrastructure

```bash
cd terraform
terraform destroy --auto-approve
```

## ✅ Code Quality & Testing

This project adheres to high code quality standards with comprehensive testing:

### Linting & Formatting
- **`make lint`:** Checks code style with `black` and `isort`
- **`make format`:** Automatically formats code with `black` and `isort`
- **Pre-commit hooks:** Automatic checks before every commit (installed via `pre-commit install`)

### Testing
- **`make test`:** Runs all `pytest` tests for unit and integration testing
- **Sample Tests:** Includes `test_unit_data.py` and `test_integration_mlflow.py` as examples
- **Examples:**
  - `test_clean_df_removes_constants_and_infs`: Verifies data cleaning logic
  - `test_mlflow_logging_fast`: Validates MLflow runs and parameter logging

### Continuous Integration
- **GitHub Actions:** `.github/workflows/ci.yml` runs `make lint`, `make test`, and `docker compose build` on every push to `main` and pull request
- **Scope:** GitHub CI is limited to code quality checks and Docker health validation
- **Pipeline Orchestration:** The complete project pipeline automation is handled by **Apache Airflow**, which manages the end-to-end ML workflow from data processing to deployment

## 📚 Dependencies

This project uses a layered dependency management approach:

- **`requirements.txt` (Project Root):** Core Python dependencies for local development, linting, formatting, and testing
- **`airflow/requirements-*.txt` (Airflow Specific):** Dependencies for the custom Airflow Docker image
  - `requirements-aws.txt`: AWS-specific packages
  - `requirements-extra.txt`: Additional Airflow operators
  - `requirements-ml.txt`: Machine learning libraries
- **`streamlit_app/requirements.txt` (Streamlit App):** Minimal dependencies for the web application serving environment

## 🔧 Key Features

- **End-to-End Automation:** Complete pipeline from data ingestion to model deployment
- **Self-Healing:** Automatic retraining triggered by data drift detection
- **Reproducible Infrastructure:** Terraform-managed AWS resources
- **Experiment Tracking:** MLflow for comprehensive experiment management
- **Real-Time Predictions:** Streamlit web application for interactive predictions
- **Quality Assurance:** Comprehensive testing and code quality checks
- **Imbalanced Data Handling:** SMOTE and advanced techniques for dealing with bankruptcy prediction challenges

## 🎯 Business Impact

This MLOps pipeline addresses critical business needs in bankruptcy prediction:

- **Risk Mitigation:** Early bankruptcy prediction enables proactive decision-making for investors and creditors
- **Automated Monitoring:** Continuous model performance tracking without manual intervention
- **Scalable Deployment:** Cloud-native architecture supports growing data volumes
- **Reproducible Results:** Consistent model training and deployment processes
- **Cost Efficiency:** Automated retraining only when necessary, reducing computational costs

**Performance Considerations:** Due to the challenging nature of the merged datasets from three different sources and the highly imbalanced distribution of bankruptcy cases, model performance may be limited. This is an expected characteristic of real-world financial prediction problems where bankruptcies are rare events, making accurate prediction inherently difficult despite advanced techniques like SMOTE.