import gc
import logging
import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


def cleanup_memory():
    """Force garbage collection and clear memory"""
    gc.collect()
    plt.close("all")


log = logging.getLogger(__name__)

# --- Configuration ---
MLFLOW_MODEL_NAME = "CorporateBankruptcyPrediction"
PROMOTION_METRIC = "recall_bankruptcy"

# Best parameters from previous runs
BEST_PARAMS = {
    "RandomForest": {
        "bootstrap": False,
        "max_depth": 14,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "min_samples_split": 5,
        "n_estimators": 106,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "XGBoost": {
        "colsample_bytree": 0.9674776711106569,
        "gamma": 0.08833152773808622,
        "learning_rate": 0.09551151648262626,
        "max_depth": 6,
        "n_estimators": 186,
        "reg_alpha": 0.03115647042203574,
        "reg_lambda": 0.5691542691392876,
        "subsample": 0.8707174795256837,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    },
}


def get_production_model_info(client, model_name):
    """Get current production model info, prioritizing aliases"""
    try:
        # Use alias-based search first, as it is the current best practice
        prod_model = client.get_model_version_by_alias(model_name, "production")
        run = client.get_run(prod_model.run_id)
        metric_value = run.data.metrics.get(PROMOTION_METRIC)
        log.info(
            f"Found production model (alias) version {prod_model.version} with {PROMOTION_METRIC}: {metric_value}"
        )
        return metric_value, prod_model
    except Exception as e:
        log.warning(f"Could not retrieve production model info from alias: {e}")
        # Fallback to stages-based search, which is now deprecated
        versions = client.search_model_versions(f"name='{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if prod_versions:
            latest_prod = max(prod_versions, key=lambda x: int(x.version))
            run = client.get_run(latest_prod.run_id)
            metric_value = run.data.metrics.get(PROMOTION_METRIC)
            log.info(
                f"Found production model (stage) version {latest_prod.version} with {PROMOTION_METRIC}: {metric_value}"
            )
            return metric_value, latest_prod
        return None, None


def promote_model_if_better(client, model_name, run_id, new_metric_value, models_dir):
    """Promote model to production if it's better than current"""
    log.info(
        f"Checking if model should be promoted (new {PROMOTION_METRIC}: {new_metric_value:.4f})"
    )

    current_metric, current_model = get_production_model_info(client, model_name)

    should_promote = current_metric is None or new_metric_value > current_metric

    if should_promote:
        log.info(f"Promoting model! Better than current: {current_metric}")

        # Register new model version
        model_uri = f"runs:/{run_id}/model"
        new_version = mlflow.register_model(model_uri, model_name)

        # Set as production using the modern alias API
        client.set_registered_model_alias(model_name, "production", new_version.version)
        log.info(f"Model version {new_version.version} set as production alias")

        # Archive old production model
        if current_model and current_model.aliases:
            # We only archive if the old model had an alias, to avoid errors
            try:
                client.delete_registered_model_alias(model_name, "production")
                log.info(
                    f"Old production alias deleted for model version {current_model.version}"
                )
            except Exception as e:
                log.warning(f"Could not delete old production alias: {e}")

        return True
    else:
        log.info(
            f"Model not promoted. Current production is better: {current_metric:.4f}"
        )
        return False


def evaluate_model(model, X_test, y_test, model_name, plots_dir):
    """Evaluate model and return metrics"""
    log.info(f"Evaluating {model_name}")

    os.makedirs(plots_dir, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    # Metrics
    metrics = {
        "f1_bankruptcy": f1_score(y_test, y_pred, pos_label=1),
        "precision_bankruptcy": precision_score(y_test, y_pred, pos_label=1),
        "recall_bankruptcy": recall_score(y_test, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0,
    }

    # Log metrics
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value)

    # Log classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("accuracy", class_report["accuracy"])

    # Generate plots
    generate_evaluation_plots(y_test, y_pred, y_prob, model_name, plots_dir)

    log.info(
        f"{model_name} - Recall: {metrics['recall_bankruptcy']:.4f}, "
        f"F1: {metrics['f1_bankruptcy']:.4f}, "
        f"ROC AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics


def generate_evaluation_plots(y_test, y_pred, y_prob, model_name, plots_dir):
    """Generate evaluation plots"""
    os.makedirs(plots_dir, exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Bankrupt", "Bankrupt"],
        yticklabels=["Non-Bankrupt", "Bankrupt"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    cm_path = os.path.join(
        plots_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # ROC Curve
    if y_prob is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)

        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        roc_path = os.path.join(
            plots_dir, f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
        )
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()


def generate_shap_analysis(model, X_train, X_test, model_name, plots_dir):
    """Generate SHAP analysis plots for tree models safely"""
    os.makedirs(plots_dir, exist_ok=True)

    try:
        log.info(f"Generating SHAP analysis for {model_name}")

        sample_size = min(5000, X_test.shape[0])
        X_test_sample = X_test.sample(n=sample_size, random_state=42).reset_index(
            drop=True
        )

        explainer = shap.TreeExplainer(model, X_train)

        # Disable strict additivity check to avoid error
        shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

        # For classification, shap_values is a list: [class0 values, class1 values]
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]  # class 1 (bankruptcy)
        else:
            shap_values_to_plot = shap_values

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_to_plot, X_test_sample, show=False, max_display=20
        )
        plt.title(f"SHAP Summary Plot - {model_name}")
        plt.tight_layout()

        summary_path = os.path.join(
            plots_dir, f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
        )
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(summary_path)
        plt.close()

        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_to_plot,
            X_test_sample,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()

        importance_path = os.path.join(
            plots_dir, f"shap_importance_{model_name.lower().replace(' ', '_')}.png"
        )
        plt.savefig(importance_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(importance_path)
        plt.close()

        log.info(f"SHAP analysis completed for {model_name}")

    except Exception as e:
        log.warning(f"Could not generate SHAP analysis for {model_name}: {e}")


def train_optimized_models(X_train, y_train, X_test, y_test, plots_dir):
    """Train models with optimized parameters"""
    models = {}
    model_metrics = {}

    os.makedirs(plots_dir, exist_ok=True)

    # Calculate scale_pos_weight for XGBoost
    neg_count = Counter(y_train)[0]
    pos_count = Counter(y_train)[1]
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    # Update XGBoost parameters
    xgb_params = BEST_PARAMS["XGBoost"].copy()
    xgb_params["scale_pos_weight"] = scale_pos_weight

    # Train Random Forest
    with mlflow.start_run(run_name="Random Forest (Optimized)", nested=True):
        rf_model = RandomForestClassifier(**BEST_PARAMS["RandomForest"])
        rf_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(BEST_PARAMS["RandomForest"])
        mlflow.log_param("model_type", "RandomForest")

        # Evaluate and log
        metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest", plots_dir)

        # Log model using the `name` parameter
        mlflow.sklearn.log_model(
            rf_model,
            "model",
            input_example=X_train.head(1),
            metadata={"source": "RandomForest-training"},
        )

        models["RandomForest"] = rf_model
        model_metrics["RandomForest"] = metrics

        current_run_id = mlflow.active_run().info.run_id
        models["RandomForest_run_id"] = current_run_id

    # Train XGBoost
    with mlflow.start_run(run_name="XGBoost (Optimized)", nested=True):
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(xgb_params)
        mlflow.log_param("model_type", "XGBoost")

        # Evaluate and log
        metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost", plots_dir)

        # Log model using the `name` parameter
        mlflow.xgboost.log_model(
            xgb_model,
            "model",
            input_example=X_train.head(1),
            metadata={"source": "XGBoost-training"},
        )

        models["XGBoost"] = xgb_model
        model_metrics["XGBoost"] = metrics

        current_run_id = mlflow.active_run().info.run_id
        models["XGBoost_run_id"] = current_run_id

    return models, model_metrics


def save_preprocessors(models_dir):
    """Save preprocessor artifacts to models directory"""
    preprocessor_files = [
        "global_scaler.pkl",
        "global_imputer.pkl",
        "feature_names.pkl",
    ]

    os.makedirs(models_dir, exist_ok=True)

    for file_name in preprocessor_files:
        file_path = os.path.join(models_dir, file_name)
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
            log.info(f"Logged {file_name} as MLflow artifact")
        else:
            log.warning(f"Preprocessor file {file_name} not found at {file_path}")


def run_model_training(data_root_dir, project_root_dir):
    """
    Main training pipeline with simplified MLflow integration
    """
    log.info("Starting simplified model training pipeline")

    # Setup directories
    models_dir = os.path.join(data_root_dir, "models")
    plots_dir = os.path.join(data_root_dir, "plots")
    mlflow_dir = os.path.join(project_root_dir, "mlflow_setup")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(mlflow_dir, exist_ok=True)

    # Setup MLflow
    mlflow_uri = f"file://{os.path.join(mlflow_dir, 'mlruns')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("CorporateBankruptcyPrediction")

    log.info(f"MLflow tracking URI: {mlflow_uri}")

    # Load data
    log.info("Loading preprocessed data")
    try:
        X_train = pd.read_csv(os.path.join(data_root_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(data_root_dir, "y_train.csv")).squeeze()
        X_test = pd.read_csv(os.path.join(data_root_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(data_root_dir, "y_test.csv")).squeeze()

        log.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

    except Exception as e:
        log.error(f"Error loading data: {e}")
        raise

    # Main training run
    with mlflow.start_run(run_name="Model Training Pipeline"):
        # Log data info
        mlflow.log_params(
            {
                "train_shape": str(X_train.shape),
                "test_shape": str(X_test.shape),
                "train_class_dist": str(dict(Counter(y_train))),
                "test_class_dist": str(dict(Counter(y_test))),
            }
        )

        # Save and log preprocessors
        save_preprocessors(models_dir)

        # Train models
        models, model_metrics = train_optimized_models(
            X_train, y_train, X_test, y_test, plots_dir
        )

        # Select best model based on recall
        best_model_name = max(
            model_metrics.keys(), key=lambda x: model_metrics[x]["recall_bankruptcy"]
        )
        best_model = models[best_model_name]
        best_metrics = model_metrics[best_model_name]
        best_run_id = models[f"{best_model_name}_run_id"]

        log.info(
            f"Best model: {best_model_name} with recall: {best_metrics['recall_bankruptcy']:.4f}"
        )

        # Generate SHAP analysis for best model
        generate_shap_analysis(best_model, X_train, X_test, best_model_name, plots_dir)

        # Model promotion
        client = MlflowClient()
        promoted = promote_model_if_better(
            client,
            MLFLOW_MODEL_NAME,
            best_run_id,
            best_metrics["recall_bankruptcy"],
            models_dir,
        )

        if promoted:
            final_model_path = os.path.join(
                models_dir, "final_best_bankruptcy_model.pkl"
            )
            joblib.dump(best_model, final_model_path)
            log.info(f"Final model saved to {final_model_path}")
        else:
            log.info("Model not promoted, not saving to models directory.")

        # Log promotion result
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("promoted_to_production", promoted)
        mlflow.log_metric("final_recall_bankruptcy", best_metrics["recall_bankruptcy"])

        cleanup_memory()

    log.info("Model training pipeline completed successfully")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 2:
        data_dir = sys.argv[1]
        project_dir = sys.argv[2]
        run_model_training(data_dir, project_dir)
    else:
        print("Usage: python script.py <data_directory> <project_directory>")
