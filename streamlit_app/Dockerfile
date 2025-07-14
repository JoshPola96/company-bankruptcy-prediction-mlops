# Stage 1: Base image with preinstalled packages (optional for future caching or local rebuild speed)
FROM python:3.9-slim-buster as base

# Install basic OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Final runtime image
FROM python:3.9-slim-buster

WORKDIR /company-bankrupt-prediction

# Copy installed packages from base stage
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application source code
COPY src/ ./src/

# Copy model files
COPY data/models/ ./data/models/

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
