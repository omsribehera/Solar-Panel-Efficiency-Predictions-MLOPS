from zenml import step
import pandas as pd
from typing import List, Tuple

@step
def split_features_target_step(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns]
    y = df[target_column]
    return X, y 