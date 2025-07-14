import logging
from datetime import datetime

from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG
from src.monitor_model import run_monitoring

log = logging.getLogger(__name__)


def decide_what_to_do_next(**kwargs):
    ti = kwargs["ti"]
    should_trigger_retrain = ti.xcom_pull(task_ids="check_for_drift")

    if should_trigger_retrain:
        log.info("✅ Drift detected — will notify and trigger retraining pipeline.")
        return "simulate_notification"
    else:
        log.info("✅ No drift detected — skipping retrain and stopping pipeline.")
        return "no_action_needed"


with DAG(
    dag_id="monitoring_pipeline_dag",
    start_date=datetime(2025, 7, 11),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "monitoring"],
    doc_md="""
    ### Monitoring Pipeline DAG

    - Checks for model/data drift daily.
    - If drift detected:
        - Notifies team (simulated).
        - Triggers downstream retraining and deployment DAGs.
    - If no drift detected:
        - Ends gracefully.
    """,
) as dag:
    check_for_drift = PythonOperator(
        task_id="check_for_drift",
        python_callable=run_monitoring,
    )

    decide_branch = BranchPythonOperator(
        task_id="decide_what_to_do_next",
        python_callable=decide_what_to_do_next,
    )

    simulate_notification = BashOperator(
        task_id="simulate_notification",
        bash_command="""
            echo "🚨🚨🚨 ALERT: Significant data drift detected! Notifying team! 🚨🚨🚨"
            echo "Drift detected on $(date)" > /opt/airflow/data/airflow_drift_notification.txt
            cat /opt/airflow/data/airflow_drift_notification.txt
        """,
    )

    trigger_data_download = TriggerDagRunOperator(
        task_id="trigger_data_download_dag",
        trigger_dag_id="data_download_bankruptcy_prediction_dag",
        wait_for_completion=True,
        reset_dag_run=True,
        conf={"source": "monitoring_pipeline_dag"},
    )

    trigger_data_processing = TriggerDagRunOperator(
        task_id="trigger_data_processing_dag",
        trigger_dag_id="data_consolidation_and_preprocessing_external_script_dag",
        wait_for_completion=True,
        reset_dag_run=True,
        conf={"source": "monitoring_pipeline_dag"},
    )

    trigger_model_training = TriggerDagRunOperator(
        task_id="trigger_model_training_dag",
        trigger_dag_id="model_training_and_evaluation_dag",
        wait_for_completion=True,
        reset_dag_run=True,
        conf={"source": "monitoring_pipeline_dag"},
    )

    trigger_deployment_pipeline = TriggerDagRunOperator(
        task_id="trigger_deployment_pipeline_dag",
        trigger_dag_id="streamlit_deployment_pipeline_dag",
        wait_for_completion=True,
        reset_dag_run=True,
        conf={"source": "monitoring_pipeline_dag"},
    )

    no_action_needed = BashOperator(
        task_id="no_action_needed",
        bash_command='echo "--- ✅ No significant data drift detected. All systems clear. ---"',
    )

    # Define workflow
    check_for_drift >> decide_branch
    decide_branch >> no_action_needed

    decide_branch >> simulate_notification
    simulate_notification >> trigger_data_download
    trigger_data_download >> trigger_data_processing
    trigger_data_processing >> trigger_model_training
    trigger_model_training >> trigger_deployment_pipeline
