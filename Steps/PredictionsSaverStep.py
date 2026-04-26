from zenml import step
import pandas as pd
import mlflow
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def predictions_saver_step(predictions: pd.Series, save_path: str = "artifacts/predictions.csv") -> None:
    try:
        predictions.to_csv(save_path, index=False)
        logging.info(f"PredictionsSaverStep: Predictions saved to {save_path}")
        mlflow.log_artifact(save_path)
        logging.info("PredictionsSaverStep: Predictions logged as MLflow artifact.")
    except Exception as e:
        logging.error(f"PredictionsSaverStep: Error: {e}")
        raise 