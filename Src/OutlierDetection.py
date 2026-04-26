import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to detect outliers in the given DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features for outlier detection.

        Returns:
        pd.DataFrame: A boolean dataframe indicating where outliers are located.
        """
        pass


# Concrete Strategy for Z-Score Based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")
        numeric_df = df.select_dtypes(include=[np.number])
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        outliers = z_scores > self.threshold
        # Reindex to match original DataFrame shape
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        outlier_mask[outliers.columns] = outliers
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outlier_mask


# Concrete Strategy for IQR Based Outlier Detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        numeric_df = df.select_dtypes(include=[np.number])
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        # Always convert outliers to DataFrame
        outliers = pd.DataFrame(outliers)
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        outlier_mask[outliers.columns] = outliers
        logging.info("Outliers detected using the IQR method.")
        return outlier_mask


# Context Class for Outlier Detection and Handling
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        outliers = self.detect_outliers(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df

        logging.info("Outlier handling completed.")
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")


# # Example usage
# if __name__ == "__main__":
#     # Example dataframe
#     df = pd.read_csv("Dataset/Extracted/train.csv")
#     df_numeric = df.select_dtypes(include=[np.number]).dropna()

#     # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
#     outlier_detector = OutlierDetector(IQROutlierDetection())

#     # Detect and handle outliers
#     outliers = outlier_detector.detect_outliers(df_numeric)
#     df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

#     print(df_cleaned.shape)
#     # Visualize outliers in specific features
#     outlier_detector.visualize_outliers(df_cleaned, features=["temperature", "irradiance"])

