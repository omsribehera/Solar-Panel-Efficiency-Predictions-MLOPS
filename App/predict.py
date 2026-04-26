import streamlit as st
import pandas as pd
import joblib
import os
import subprocess
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SystemConfig

# Configure logging
logging.basicConfig(
    filename="App/app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def show_predict_tab():
    """
    Display the Prediction tab in the Streamlit app.
    Uses the ZenML inference pipeline for robust predictions.
    """
    st.header("Model Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])
    if uploaded_file is not None:
        # Save uploaded file
        temp_path = "App/uploaded.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logging.info(f"Prediction: File uploaded: {uploaded_file.name}")
        st.subheader("Input Data Preview")
        df = pd.read_csv(temp_path)
        st.dataframe(df.head())

        # Run the inference pipeline
        config = SystemConfig()
        model_import_path = "joblib:load"
        feature_columns = config.feature_columns
        try:
            logging.info("Prediction: Running inference pipeline.")
            result = subprocess.run([
                "python", "run_inference.py",
                "--model_path", "artifacts/model.joblib",
                "--data_path", temp_path,
                "--feature_columns", ",".join(feature_columns),
                "--model_import_path", model_import_path
            ], check=True)
            # After pipeline run, load predictions
            pred_path = "artifacts/predictions.csv"
            if os.path.exists(pred_path):
                preds_df = pd.read_csv(pred_path)
                st.subheader("Predictions")
                st.dataframe(preds_df)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=preds_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                logging.info("Prediction: Predictions displayed and download available.")
            else:
                logging.error("Prediction: Predictions file not found after pipeline run.")
                st.error("Prediction failed or predictions file not found.")
        except Exception as e:
            logging.error(f"Prediction: Error running inference pipeline: {e}")
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Please upload a CSV file to get predictions.") 