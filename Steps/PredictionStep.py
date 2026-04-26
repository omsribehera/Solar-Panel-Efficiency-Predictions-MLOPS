from zenml import step
import pandas as pd
from typing import Any
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def prediction_step(model: Any, X_test: pd.DataFrame) -> pd.Series:
    """
    Make predictions using the trained model.
    """
    try:
        logging.info(f"PredictionStep: Making predictions for input shape {X_test.shape}")
        predictions = model.predict(X_test)
        logging.info(f"PredictionStep: Predictions made, output shape {len(predictions)}")
        return pd.Series(predictions)
    except Exception as e:
        logging.error(f"PredictionStep: Error: {e}")
        raise
