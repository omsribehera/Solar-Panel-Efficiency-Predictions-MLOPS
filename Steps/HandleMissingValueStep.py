from zenml import step
import pandas as pd
from typing import Optional, List
from Src.HandleMissingValues import (
    MissingValueHandler,
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    KNNImputationStrategy,
    StatisticalModelImputationStrategy,
    CategoricalMissingFeatureStrategy
)
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def handle_missing_value_step(
    df: pd.DataFrame,
    strategy: str = 'catboost',
    columns: Optional[List[str]] = None,
    fill_value: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame using the specified strategy.
    """
    try:
        logging.info(f"HandleMissingValueStep: Strategy={strategy}, Columns={columns}, FillValue={fill_value}")
        if strategy == 'drop':
            handler = MissingValueHandler(DropMissingValuesStrategy())
        elif strategy == 'mean':
            handler = MissingValueHandler(FillMissingValuesStrategy(method='mean'))
        elif strategy == 'median':
            handler = MissingValueHandler(FillMissingValuesStrategy(method='median'))
        elif strategy == 'mode':
            handler = MissingValueHandler(FillMissingValuesStrategy(method='mode'))
        elif strategy == 'constant':
            handler = MissingValueHandler(FillMissingValuesStrategy(method='constant', fill_value=fill_value))
        elif strategy == 'knn':
            handler = MissingValueHandler(KNNImputationStrategy())
        elif strategy == 'catboost':
            handler = MissingValueHandler(StatisticalModelImputationStrategy(model_type='catboost', iterations=100, learning_rate=0.01))
        elif strategy == 'categorical':
            if not columns or not fill_value:
                logging.error('HandleMissingValueStep: For categorical strategy, columns and fill_value must be provided')
                raise ValueError('For categorical strategy, columns and fill_value must be provided')
            handler = MissingValueHandler(CategoricalMissingFeatureStrategy(column_name=columns[0], fill_value=fill_value))
        else:
            logging.error(f'HandleMissingValueStep: Unknown strategy: {strategy}')
            raise ValueError(f'Unknown strategy: {strategy}')
        result = handler.handle_missing_values(df, columns=columns)
        logging.info("HandleMissingValueStep: Missing value handling successful.")
        return result
    except Exception as e:
        logging.error(f"HandleMissingValueStep: Error: {e}")
        raise
