# Stage 1: ML base image to build wheels only (optional caching)
FROM python:3.12-slim AS ml-base

COPY requirements-ml.txt /requirements/

# Build wheels (optional, if you want to cache)
RUN pip wheel --wheel-dir=/wheels -r /requirements/requirements-ml.txt


# Stage 2: Final Airflow image
FROM apache/airflow:3.0.2

USER root

# Install OS utilities and Terraform
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip curl gnupg software-properties-common && \
    curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/hashicorp.list && \
    apt-get update && \
    apt-get install -y terraform && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Prepare kaggle directory
RUN mkdir -p /home/airflow/.kaggle && chmod 700 /home/airflow/.kaggle

USER airflow

# Copy requirements
COPY requirements-ml.txt requirements-aws.txt requirements-extra.txt /requirements/

# Copy prebuilt wheels (optional)
COPY --from=ml-base /wheels /wheels

# Install ML requirements (from wheels if available)
RUN pip install --no-cache-dir --find-links=/wheels -r /requirements/requirements-ml.txt

# Install AWS-related packages
RUN pip install --no-cache-dir -r /requirements/requirements-aws.txt

# Install additional utilities
RUN pip install --no-cache-dir -r /requirements/requirements-extra.txt

# Optional additional requirements
ARG _PIP_ADDITIONAL_REQUIREMENTS=""
RUN if [ -n "$_PIP_ADDITIONAL_REQUIREMENTS" ]; then pip install --no-cache-dir $_PIP_ADDITIONAL_REQUIREMENTS; fi
