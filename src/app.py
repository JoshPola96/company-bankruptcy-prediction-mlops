import json  # For S3 logging
import logging
import os
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd
import streamlit as st

# Configure logging for the Streamlit app
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data", "models"))

SCALER_PATH = os.path.join(ARTIFACTS_DIR, "global_scaler.pkl")
IMPUTER_PATH = os.path.join(ARTIFACTS_DIR, "global_imputer.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "final_best_bankruptcy_model.pkl")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")


@st.cache_resource
def load_artifacts():
    """Loads the latest MLflow model and preprocessing artifacts."""
    try:
        # Load preprocessing objects
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        # Load feature names dynamically from the saved pickle file
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        model = joblib.load(MODEL_PATH)

        log.info("Tuned model and preprocessing artifacts loaded successfully.")
        return model, scaler, imputer, feature_names
    except Exception as e:
        st.error(f"Error loading model or artifacts: {e}")
        log.error(f"Loading error: {e}")
        return None, None, None, None


model, scaler, imputer, feature_names = load_artifacts()

st.set_page_config(page_title="Company Bankruptcy Prediction", layout="wide")

# Custom CSS for improved aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #28a745; /* Green */
        color: white;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #218838; /* Darker green */
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 10px 15px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    }
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        font-size: 16px;
    }
    .stAlert.stAlert--success {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    .stAlert.stAlert--error {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    .stAlert.stAlert--info {
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
    }
    .header-style {
        color: #2c3e50;
        text-align: center;
        font-size: 3em;
        margin-bottom: 25px;
        font-weight: 700;
    }
    .subheader-style {
        color: #34495e;
        font-size: 1.8em;
        margin-top: 30px;
        margin-bottom: 15px;
        font-weight: 600;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 5px;
    }
    .stSidebar {
        background-color: #34495e; /* Darker background for better contrast */
        border-right: 1px solid #2c3e50; /* Darker border */
        padding: 20px;
        border-radius: 10px;
        color: white; /* Ensure text is readable on new background */
    }
    .stSidebar .st-emotion-cache-1lcbm9l { /* Targeting specific Streamlit element for sidebar text */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="header-style">Company Bankruptcy Prediction</h1>',
    unsafe_allow_html=True,
)
st.write(
    "Enter the financial attributes below to predict the likelihood of a company going bankrupt."
)

if model is None or scaler is None or imputer is None or feature_names is None:
    st.error(
        "Model artifacts could not be loaded. Please ensure data processing and model training DAGs have run and saved artifacts to the correct location."
    )
else:
    # --- Input Fields (Dynamic based on feature_names) ---
    st.markdown(
        '<h2 class="subheader-style">Financial Attributes Input</h2>',
        unsafe_allow_html=True,
    )

    # Message about the most impactful features
    st.info(
        "Based on SHAP analysis, 'AfterTax_Net_Interest_Rate' and 'Accounts_Receivable_Turnover' are often the most impactful features for predictions."
    )

    # Create columns for better layout (e.g., 3 columns)
    num_cols = 3
    cols = st.columns(num_cols)
    input_features = {}

    for i, feature_name in enumerate(feature_names):
        with cols[i % num_cols]:
            # All features are adjustable and default to 0.0
            input_features[feature_name] = st.number_input(
                f"{feature_name}", value=0.0, format="%.4f", key=f"input_{feature_name}"
            )

    # --- Prediction ---
    st.markdown("---")
    if st.button("Predict Bankruptcy"):
        # Create a DataFrame from the input features
        input_df = pd.DataFrame([input_features])

        # Ensure input DataFrame has the same columns and order as training data
        # This is crucial because the model expects features in a specific order
        input_df = input_df.reindex(columns=feature_names, fill_value=np.nan)

        # Apply the same preprocessing steps as the training pipeline
        # 1. Imputation (using the loaded imputer)
        input_imputed = imputer.transform(input_df)
        input_imputed_df = pd.DataFrame(
            input_imputed, columns=feature_names, index=input_df.index
        )

        # 2. Scaling (using the loaded scaler)
        input_scaled = scaler.transform(input_imputed_df)
        input_scaled_df = pd.DataFrame(
            input_scaled, columns=feature_names, index=input_df.index
        )

        try:
            # The loaded model is a Pipeline, so it handles PCA internally
            prediction_proba = model.predict_proba(input_scaled_df)[:, 1][
                0
            ]  # Probability of bankruptcy (class 1)
            prediction_class = (
                1 if prediction_proba > 0.5 else 0
            )  # Simple threshold for binary classification

            st.markdown(
                '<h2 class="subheader-style">Prediction Results:</h2>',
                unsafe_allow_html=True,
            )
            st.write(f"Probability of Bankruptcy: **{prediction_proba:.4f}**")
            if prediction_class == 1:
                st.error("Prediction: **Likely to go Bankrupt** 📉")
            else:
                st.success("Prediction: **Unlikely to go Bankrupt** ✅")

            # --- For Evidently Monitoring (Conceptual) ---
            # This part assumes you have AWS credentials configured for boto3
            # and an S3 bucket named "company-bankruptcy-prediction-monitoring"
            # created and accessible.
            # --- For Evidently Monitoring (Using EC2 Instance Metadata Service) ---
            try:
                import boto3
                from botocore.exceptions import ClientError, NoCredentialsError

                # Create boto3 session that will automatically use EC2 instance metadata
                # This works when running on EC2 with IAM role attached
                session = boto3.Session()
                s3_client = session.client("s3")

                MONITORING_BUCKET = "company-bankruptcy-prediction-monitoring"

                # Test S3 access by trying to head the bucket
                try:
                    s3_client.head_bucket(Bucket=MONITORING_BUCKET)
                    st.info("✅ S3 bucket access confirmed")
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        st.error(f"❌ S3 bucket '{MONITORING_BUCKET}' not found")
                        raise
                    else:
                        st.error(f"❌ S3 access error: {e}")
                        raise

                inference_log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "input_features": input_features,
                    "predicted_proba": prediction_proba,
                    "predicted_class": prediction_class,
                }

                object_key = f"inference_logs/{datetime.now().strftime('%Y-%m-%d')}/{datetime.now().isoformat()}-{np.random.randint(0,10000)}.json"
                s3_client.put_object(
                    Bucket=MONITORING_BUCKET,
                    Key=object_key,
                    Body=json.dumps(inference_log_data),
                )
                st.success(f"✅ Inference data logged to S3: `{object_key}`")
                log.info(f"Inference data logged to S3: {object_key}")

            except ImportError:
                st.warning("⚠️ Boto3 not installed. Install with `pip install boto3`")
                log.warning("Boto3 not installed")
            except NoCredentialsError:
                st.error(
                    "❌ AWS credentials not found. Ensure EC2 instance has IAM role attached."
                )
                log.error("AWS credentials not found")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "AccessDenied":
                    st.error("❌ Access denied to S3. Check IAM role permissions.")
                else:
                    st.error(f"❌ S3 Client Error: {e}")
                log.error(f"S3 ClientError: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error logging to S3: {str(e)}")
                log.error(f"Unexpected S3 error: {e}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            log.error(f"Prediction error: {e}")

st.sidebar.header("About This App")
st.sidebar.info(
    "This application predicts company bankruptcy based on financial metrics. It uses a machine learning model trained as part of the MLOps Zoomcamp final project."
)
st.sidebar.text("Developed for MLOps Zoomcamp Final Project")
