.PHONY: setup lint format test mlflow_up mlflow_down airflow_init airflow_up airflow_down airflow_restart airflow_logs clean all_up all_down

SHELL := /bin/bash

# MLflow
MLFLOW_BACKEND_URI = ./mlflow_setup/mlruns
MLFLOW_ARTIFACT_ROOT = ./mlflow_setup/mlflow_artifacts
MLFLOW_HOST = 0.0.0.0
MLFLOW_PORT = 5000

# Local venv
VENV_NAME = venv

## ---------- Local development ----------

setup:
	@echo "🔧 Setting up local Python virtual environment..."
	@python3 -m venv $(VENV_NAME)
	@source $(VENV_NAME)/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt
	@echo "✅ Local setup complete. Activate with: source $(VENV_NAME)/bin/activate"

lint: setup
	@echo "🔎 Running code lint checks..."
	@source $(VENV_NAME)/bin/activate && \
	black --check src airflow/dags
	@source $(VENV_NAME)/bin/activate && \
	isort --check-only src airflow/dags

format: setup
	@echo "🎨 Auto-formatting code..."
	@source $(VENV_NAME)/bin/activate && \
	isort src airflow/dags
	@source $(VENV_NAME)/bin/activate && \
	black src airflow/dags

test: setup
	@echo "🧪 Running tests..."
	@source $(VENV_NAME)/bin/activate && \
	DATA_ROOT_DIR=$(shell pwd)/data pytest tests/

## ---------- MLflow ----------

mlflow_up:
	@echo "🚀 Starting MLflow Tracking Server..."
	@mkdir -p $(MLFLOW_BACKEND_URI) $(MLFLOW_ARTIFACT_ROOT)
	@mlflow server \
		--backend-store-uri $(MLFLOW_BACKEND_URI) \
		--default-artifact-root $(MLFLOW_ARTIFACT_ROOT) \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) &
	@sleep 5
	@echo "✅ MLflow server started at: http://localhost:$(MLFLOW_PORT)"

mlflow_down:
	@echo "🛑 Stopping MLflow Tracking Server..."
	@pkill -f "mlflow server" || true
	@echo "✅ MLflow server stopped."

## ---------- Airflow (Docker) ----------

airflow_init:
	@echo "⚙️ Initializing Airflow database..."
	@docker compose --env-file airflow/.env --project-directory airflow up airflow-init

airflow_up:
	@echo "🚀 Starting Airflow services..."
	@docker compose --env-file airflow/.env --project-directory airflow up -d --build
	@echo "✅ Airflow services started!"
	@echo "💻 Access Airflow UI at: http://localhost:8080"

airflow_down:
	@echo "🛑 Stopping Airflow services and removing volumes..."
	@docker compose --env-file airflow/.env --project-directory airflow down -v

airflow_restart: airflow_down airflow_up

airflow_logs:
	@docker compose --env-file airflow/.env --project-directory airflow logs -f

## ---------- Cleanup ----------

clean:
	@echo "🧹 Cleaning up generated and build files..."
	@rm -rf $(VENV_NAME)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@rm -rf .pytest_cache .ipynb_checkpoints
	@echo "✅ Cleanup complete."

## ---------- All-in-one ----------

all_up: format lint test airflow_up mlflow_up
	@echo "🎉 ✅ All checks done! Airflow and MLflow are up and running."
	@echo "💻 Access Airflow UI at: http://localhost:8080"
	@echo "💻 Access MLflow UI at: http://localhost:$(MLFLOW_PORT)"

all_down: airflow_down mlflow_down
	@echo "🛑 All services stopped and volumes removed."
