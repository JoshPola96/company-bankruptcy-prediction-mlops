import os

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_mlflow_logging_fast(monkeypatch):
    """
    Fast test to ensure MLflow logging logic works without full training.
    """

    # Clear possible global env vars
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_ARTIFACT_URI", None)

    tracking_dir = "./mlruns_test"
    artifact_dir = "./mlruns_test_artifacts"

    os.makedirs(artifact_dir, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{tracking_dir}")

    experiment_name = "TestDummyExperiment"

    # Create experiment with explicit artifact location
    try:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f"file:{artifact_dir}",
        )
    except mlflow.exceptions.MlflowException:
        # Already exists
        pass

    mlflow.set_experiment(experiment_name)

    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)

    with mlflow.start_run() as run:
        model = LogisticRegression()
        model.fit(X, y)
        mlflow.log_param("dummy_param", 123)
        mlflow.sklearn.log_model(model, artifact_path="model")

    client = mlflow.tracking.MlflowClient()
    run_id = run.info.run_id
    run_data = client.get_run(run_id)

    import shutil

    shutil.rmtree(tracking_dir, ignore_errors=True)
    shutil.rmtree(artifact_dir, ignore_errors=True)

    assert (
        run_data.data.params["dummy_param"] == "123"
    ), "MLflow param not logged properly"
