from zenml import step
import pandas as pd
from typing import Any
from Src.ModelBuilding import (
    ModelBuilder,
    LinearRegressionStrategy,
    RandomForestStrategy,
    XGBoostStrategy,
    CatBoostStrategy
)
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest'
) -> Any:
    """
    Build and train a regression model using the specified strategy.
    """
    try:
        logging.info(f"ModelBuildingStep: Model type={model_type}")
        if model_type == 'linear':
            builder = ModelBuilder(LinearRegressionStrategy())
        elif model_type == 'random_forest':
            builder = ModelBuilder(RandomForestStrategy())
        elif model_type == 'xgboost':
            builder = ModelBuilder(XGBoostStrategy())
        elif model_type == 'catboost':
            builder = ModelBuilder(CatBoostStrategy())
        else:
            logging.error(f'Unknown model_type: {model_type}')
            raise ValueError(f'Unknown model_type: {model_type}')
        model = builder.build_model(X_train, y_train)
        logging.info("ModelBuildingStep: Model training successful.")
        return model
    except Exception as e:
        logging.error(f"ModelBuildingStep: Error: {e}")
        raise
