import logging
import os
from datetime import datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

log = logging.getLogger(__name__)

DATA_ROOT_DIR = "/opt/airflow/data"

DATASETS = [
    {
        "name": "polish_companies",
        "url": "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip",
        "zip_filename": "company-bankruptcy-prediction-dataset-polish.zip",
        "extracted_csv": None,
        "final_csv": None,
        "download_method": "curl",
    },
    {
        "name": "us_companies",
        "kaggle_dataset": "utkarshx27/american-companies-bankruptcy-prediction-dataset",
        "zip_filename": "company-bankruptcy-prediction-dataset-us.zip",
        "extracted_csv": "american_bankruptcy.csv",
        "final_csv": "company-bankruptcy-prediction-dataset-us.csv",
        "download_method": "kaggle",
    },
    {
        "name": "taiwan_companies",
        "kaggle_dataset": "fedesoriano/company-bankruptcy-prediction",
        "zip_filename": "company-bankruptcy-prediction-dataset-taiwan.zip",
        "extracted_csv": "data.csv",
        "final_csv": "company-bankruptcy-prediction-dataset-taiwan.csv",
        "download_method": "kaggle",
    },
]

with DAG(
    dag_id="data_download_bankruptcy_prediction_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bankruptcy", "data_ingestion", "etl"],
    doc_md="""
    ### Data Ingestion DAG for Bankruptcy Prediction Datasets

    Downloads datasets, unzips directly into data folder, renames CSVs to standard names, and deletes zip files.
    """,
) as dag:
    for dataset in DATASETS:
        zip_file_path = os.path.join(DATA_ROOT_DIR, dataset["zip_filename"])

        # Download task
        if dataset["download_method"] == "curl":
            download_task = BashOperator(
                task_id=f'download_{dataset["name"]}',
                bash_command=f'curl -L -o {zip_file_path} "{dataset["url"]}"',
                do_xcom_push=False,
            )
        elif dataset["download_method"] == "kaggle":
            download_task = BashOperator(
                task_id=f'download_{dataset["name"]}',
                bash_command=f"""
                export KAGGLE_CONFIG_DIR=/home/airflow/.kaggle &&
                cd {DATA_ROOT_DIR} &&
                kaggle datasets download -d {dataset["kaggle_dataset"]} &&
                mv {dataset["kaggle_dataset"].split("/")[1]}.zip {dataset["zip_filename"]}
                """,
                do_xcom_push=False,
            )

        # Unzip task
        unzip_task = BashOperator(
            task_id=f'unzip_{dataset["name"]}',
            bash_command=f"unzip -o {zip_file_path} -d {DATA_ROOT_DIR}",
            do_xcom_push=False,
        )

        # Rename CSV task (only if final_csv is specified)
        if dataset["final_csv"] and dataset["extracted_csv"]:
            src_csv = os.path.join(DATA_ROOT_DIR, dataset["extracted_csv"])
            dst_csv = os.path.join(DATA_ROOT_DIR, dataset["final_csv"])
            rename_task = BashOperator(
                task_id=f'rename_{dataset["name"]}',
                bash_command=(
                    f'if [ -f "{src_csv}" ]; then mv "{src_csv}" "{dst_csv}"; '
                    f'else echo "Expected CSV not found: {src_csv}"; exit 1; fi'
                ),
                do_xcom_push=False,
            )
        else:
            rename_task = None

        # Delete zip task
        delete_task = BashOperator(
            task_id=f'delete_zip_{dataset["name"]}',
            bash_command=f"rm -f {zip_file_path}",
            do_xcom_push=False,
        )

        # Dependencies
        download_task >> unzip_task
        if rename_task:
            unzip_task >> rename_task >> delete_task
        else:
            unzip_task >> delete_task
