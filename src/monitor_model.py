import json
import logging

import boto3
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

S3_BUCKET = "company-bankruptcy-prediction-monitoring"

# --- Configurable options ---
MIN_SAMPLES = 30
DRIFT_THRESHOLD = 0.6
IMPORTANT_COLUMNS = [
    "AfterTax_Net_Interest_Rate",
    "Accounts_Receivable_Turnover",
    # Add any other important columns you'd like to focus on
]


def load_reference_data(file_path):
    log.info(f"Loading reference data from {file_path}...")
    df = pd.read_csv(file_path)
    return df


def get_inference_logs_from_s3(days_ago=1):
    s3 = boto3.client("s3")
    current_date = pd.Timestamp.now(tz="UTC").date()
    start_date = current_date - pd.Timedelta(days=days_ago - 1)

    all_data = []
    for day_offset in range(days_ago):
        date_str = (start_date + pd.Timedelta(days=day_offset)).isoformat()
        prefix = f"inference_logs/{date_str}/"
        log.info(f"Searching for logs with prefix: {prefix}")

        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if "Contents" in response:
            for item in response["Contents"]:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=item["Key"])
                log_data = json.loads(obj["Body"].read())
                all_data.append(log_data)

    log.info(f"Collected {len(all_data)} inference logs.")
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data)


def run_monitoring():
    reference_data_path = "/opt/airflow/data/X_train.csv"
    reference_df = load_reference_data(reference_data_path)
    current_df = get_inference_logs_from_s3(days_ago=1)

    if current_df.empty or current_df.shape[0] < MIN_SAMPLES:
        log.warning(
            f"Too few samples ({current_df.shape[0]}). Skipping drift detection."
        )
        return False

    # Convert JSON feature columns to DataFrame
    current_features_df = pd.DataFrame(current_df["input_features"].tolist())

    # --- Option: Drop columns with no variation ---
    non_constant_cols = [
        col
        for col in current_features_df.columns
        if current_features_df[col].nunique() > 1
    ]
    current_features_df = current_features_df[non_constant_cols]
    reference_df = reference_df[non_constant_cols]
    log.info(
        f"Columns remaining after removing constant columns: {len(non_constant_cols)}"
    )

    # --- Option: Keep only important columns (partial columns) ---
    if IMPORTANT_COLUMNS:
        selected_cols = list(set(IMPORTANT_COLUMNS) & set(current_features_df.columns))
        if selected_cols:
            current_features_df = current_features_df[selected_cols]
            reference_df = reference_df[selected_cols]
            log.info(
                f"Columns kept after filtering to important columns: {selected_cols}"
            )
        else:
            log.warning(
                "No overlap with important columns. Using all columns left after previous filtering."
            )

    # --- Option: Drop columns with near-zero variance ---
    variance_cols = [
        col
        for col in current_features_df.columns
        if current_features_df[col].std() > 1e-5
    ]
    current_features_df = current_features_df[variance_cols]
    reference_df = reference_df[variance_cols]
    log.info(f"Columns kept after filtering by variance: {variance_cols}")

    if len(variance_cols) == 0:
        log.warning("No columns left after filtering. Skipping drift detection.")
        return False

    # Separate numeric vs categorical
    numeric_columns = current_features_df.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_columns = list(set(current_features_df.columns) - set(numeric_columns))

    data_definition = DataDefinition(
        numerical_columns=numeric_columns, categorical_columns=categorical_columns
    )

    ref_dataset = Dataset.from_pandas(reference_df, data_definition=data_definition)
    curr_dataset = Dataset.from_pandas(
        current_features_df, data_definition=data_definition
    )

    report = Report(metrics=[DriftedColumnsCount()])
    snapshot = report.run(reference_data=ref_dataset, current_data=curr_dataset)
    result = snapshot.dict()

    drifted_columns = result["metrics"][0]["value"]["count"]
    share_drifted_columns = result["metrics"][0]["value"]["share"]

    log.info(f"Number of drifted columns: {drifted_columns}")
    log.info(f"Share of drifted columns: {share_drifted_columns}")

    dataset_drift = share_drifted_columns > DRIFT_THRESHOLD

    # Save HTML report to local file
    local_html_path = "/opt/airflow/data/evidently_report.html"
    snapshot.save_html(local_html_path)

    # Upload to S3
    report_key = (
        f"monitoring_reports/data_drift_report_{pd.Timestamp.now().isoformat()}.html"
    )
    s3_client = boto3.client("s3")
    with open(local_html_path, "rb") as f:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=report_key,
            Body=f,
            ContentType="text/html",
        )
    log.info(
        f"Monitoring report saved to S3. View at: https://{S3_BUCKET}.s3.amazonaws.com/{report_key}"
    )

    if dataset_drift:
        log.info("Significant dataset-level drift detected.")
        return True
    else:
        log.info("No significant dataset-level drift detected.")
        return False


if __name__ == "__main__":
    if run_monitoring():
        log.info(
            "Significant data drift detected! Retraining or notification workflow should be triggered."
        )
    else:
        log.info("No significant data drift detected. All clear.")
