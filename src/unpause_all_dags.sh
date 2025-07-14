#!/bin/bash

echo "🔓 Starting to unpause downstream DAGs..."

# List of DAG IDs
dags=(
  "data_download_bankruptcy_prediction_dag"
  "data_consolidation_and_preprocessing_external_script_dag"
  "model_training_and_evaluation_dag"
  "streamlit_deployment_pipeline_dag"
)

for dag_id in "${dags[@]}"; do
  echo "🔓 Unpausing DAG: $dag_id"
  airflow dags unpause "$dag_id"
done

echo "✅ All specified DAGs have been unpaused."
