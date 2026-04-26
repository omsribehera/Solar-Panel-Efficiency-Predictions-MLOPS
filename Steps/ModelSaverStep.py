from zenml import step
import joblib
from typing import Any
import os
import mlflow
import numpy as np
import pandas as pd
from mlflow.sklearn import log_model
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def model_saver_step(model: Any, save_path: str = "artifacts/model.joblib") -> None:
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        logging.info(f"ModelSaverStep: Model saved to {save_path}")
        input_example = None
        if hasattr(model, 'feature_names_in_'):
            input_example = pd.DataFrame([np.zeros(len(model.feature_names_in_))], columns=model.feature_names_in_)
        if input_example is not None:
            log_model(model, "model", input_example=input_example)
            logging.info("ModelSaverStep: Model logged to MLflow with input example.")
        else:
            log_model(model, "model")
            logging.info("ModelSaverStep: Model logged to MLflow without input example.")
    except Exception as e:
        logging.error(f"ModelSaverStep: Error: {e}")
        raise 