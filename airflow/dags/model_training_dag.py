import os
from datetime import datetime

from airflow.operators.python import PythonOperator

from airflow import DAG

# Project root and data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, "data")


def wrapped_training_callable(**context):
    import os
    import sys

    # Get parameters from context
    data_root_dir = context.get("data_root_dir", DATA_ROOT_DIR)
    project_root_dir = context.get("project_root_dir", PROJECT_ROOT)

    # Validate paths exist
    if not os.path.exists(data_root_dir):
        raise FileNotFoundError(f"Data root directory does not exist: {data_root_dir}")

    # Check for required data files
    required_files = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(data_root_dir, f))
    ]

    if missing_files:
        raise FileNotFoundError(f"Missing required data files: {missing_files}")

    # Add src to path
    src_path = os.path.join(project_root_dir, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

    try:
        from train_model import run_model_training

        return run_model_training(data_root_dir, project_root_dir)
    except ImportError as e:
        raise ImportError(f"Failed to import run_model_training: {e}")


# In your DAG definition:
with DAG(
    dag_id="model_training_and_evaluation_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bankruptcy", "model_training", "mlops", "mlflow", "pca"],
) as dag:
    train_evaluate_models_task = PythonOperator(
        task_id="train_and_evaluate_models",
        python_callable=wrapped_training_callable,
        op_kwargs={
            "data_root_dir": DATA_ROOT_DIR,
            "project_root_dir": PROJECT_ROOT,
        },
        # execution_timeout=timedelta(hours=3),  # Add timeout
    )
