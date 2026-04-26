from zenml import step
import pandas as pd
from Src.FeatureEngineering import CategoricalFeatureEncoder
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering: encode categorical features and create new features.
    """
    # Example: encode 'error_code' and 'installation_type', then create 'power' and 'area'
    encoder = CategoricalFeatureEncoder(encoding_type='one_hot')
    if 'error_code' in df.columns:
        df = encoder.encode_feature(df, 'error_code')
        logging.info("FeatureEngineeringStep: Encoded 'error_code' feature.")
        if 'error_code_is_missing' in df.columns:
            df = df.drop('error_code_is_missing', axis=1)
    if 'installation_type' in df.columns:
        df = encoder.encode_feature(df, 'installation_type')
        logging.info("FeatureEngineeringStep: Encoded 'installation_type' feature.")
    # Create new features if columns exist
    if 'voltage' in df.columns and 'current' in df.columns:
        df['power'] = df['voltage'] * df['current']
        logging.info("FeatureEngineeringStep: Created 'power' feature.")
    if 'power' in df.columns and 'efficiency' in df.columns:
        df['area'] = df['power'] / df['efficiency']
        logging.info("FeatureEngineeringStep: Created 'area' feature.")
    return df
