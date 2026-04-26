from zenml import step
import pandas as pd
from typing import Any, Dict
from Src.ModelEvaluator import ModelEvaluator, RegressionModelEvaluationStrategy
import mlflow
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def model_evaluation_step(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    try:
        logging.info("ModelEvaluationStep: Starting model evaluation.")
        evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
        metrics = evaluator.evaluate(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            logging.info(f"ModelEvaluationStep: Metric {k}={v}")
        logging.info("ModelEvaluationStep: Model evaluation successful.")
        return metrics
    except Exception as e:
        logging.error(f"ModelEvaluationStep: Error: {e}")
        raise 