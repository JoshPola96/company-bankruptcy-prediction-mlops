import os
import sys
from datetime import datetime

from airflow.operators.python import PythonOperator

from airflow import DAG

# Dynamically determine project root (assuming this DAG file is inside 'dags/')
# This PROJECT_ROOT will be /opt/airflow inside the container
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to 'src' inside the container
SRC_PATH = os.path.join(PROJECT_ROOT, "src")  # This resolves to /opt/airflow/src

# Add to sys.path so we can import
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Define the data root directory *as it exists inside the Docker container*
# This is the crucial part that must be consistent with your docker-compose.yaml volume mount
# Host: ../data  --> Container: /opt/airflow/data
CONTAINER_DATA_ROOT_DIR = os.path.join(
    PROJECT_ROOT, "data"
)  # This is /opt/airflow/data

# Remove the old DATA_ROOT_DIR = os.path.expanduser('~/company-bankrupt-prediction/data') line
# This was causing confusion by potentially overriding or being accidentally used.


def wrapped_data_preprocessing_callable(
    data_root_dir_param,
):  # Use a distinct parameter name
    """
    Wrapper function so that import happens at runtime rather than parse time.
    """
    from data_processing import run_data_processing

    # Pass the received parameter directly to the processing function
    return run_data_processing(data_root_dir_param)


with DAG(
    dag_id="data_consolidation_and_preprocessing_external_script_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bankruptcy", "data_prep", "mlops", "smote", "external_script"],
    doc_md="""
    ### Data Consolidation and Preprocessing DAG (External Script)

    This DAG orchestrates a comprehensive data cleaning, harmonization,
    imputation, scaling, train-test splitting, and class imbalance handling
    using SMOTE. The core logic is housed entirely in the external Python script
    `src/data_processing.py`, making the DAG lean and focused on orchestration.

    **Dependencies:**
    - The data in `data/extracted_datasets/` (e.g., from a `data_ingestion` DAG)
      must be present before this DAG runs.
    """,
) as dag:
    run_data_preprocessing_task = PythonOperator(
        task_id="run_data_consolidation_and_preprocessing_script",
        python_callable=wrapped_data_preprocessing_callable,
        op_kwargs={
            "data_root_dir_param": CONTAINER_DATA_ROOT_DIR,  # Pass the container's data path here
        },
    )
