from zenml import step
import pandas as pd
from typing import Optional
from Src.OutlierDetection import (
    OutlierDetector,
    ZScoreOutlierDetection,
    IQROutlierDetection
)
import logging

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def outlier_detection_step(
    df: pd.DataFrame,
    method: str = 'iqr',
    handle: str = 'remove'
) -> pd.DataFrame:
    """
    Detect and handle outliers in the DataFrame using the specified method.
    """
    try:
        logging.info(f"OutlierDetectionStep: Method={method}, Handle={handle}")
        if method == 'zscore':
            detector = OutlierDetector(ZScoreOutlierDetection())
        elif method == 'iqr':
            detector = OutlierDetector(IQROutlierDetection())
        else:
            logging.error(f'OutlierDetectionStep: Unknown outlier detection method: {method}')
            raise ValueError(f'Unknown outlier detection method: {method}')
        result = detector.handle_outliers(df, method=handle)
        logging.info("OutlierDetectionStep: Outlier handling successful.")
        return result
    except Exception as e:
        logging.error(f"OutlierDetectionStep: Error: {e}")
        raise
