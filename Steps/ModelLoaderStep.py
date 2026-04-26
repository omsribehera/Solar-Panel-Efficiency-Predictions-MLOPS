from zenml import step
import joblib
from typing import Any
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def model_loader_step(model_path: str) -> Any:
    """
    Load a pre-trained model from the given file path.
    """
    try:
        logging.info(f"ModelLoaderStep: Loading model from {model_path}")
        model = joblib.load(model_path)
        logging.info("ModelLoaderStep: Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"ModelLoaderStep: Error: {e}")
        raise
