import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CategoricalFeatureEncoder:
    def __init__(self, encoding_type: str = 'one_hot'):
        """
        Initializes the CategoricalFeatureEncoder.

        Parameters:
        encoding_type (str): Type of encoding to use ('one_hot', 'label', or 'ordinal')
        """
        self.encoding_type = encoding_type
        self.encoders = {}
        self.feature_names = {}
        self.missing_indicators = {}

    def encode_feature(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """
        Encodes a categorical feature and its missing values.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        feature_name (str): Name of the categorical feature to encode

        Returns:
        pd.DataFrame: DataFrame with encoded features
        """
        if feature_name not in df.columns:
            raise ValueError(f"Feature {feature_name} not found in DataFrame")

        df_encoded = df.copy()
        
        # Create missing value indicator
        missing_indicator_name = f"{feature_name}_is_missing"
        df_encoded[missing_indicator_name] = df[feature_name].isnull().astype(int)
        self.missing_indicators[feature_name] = missing_indicator_name

        # Handle the categorical values
        if self.encoding_type == 'one_hot':
            df_encoded = self._one_hot_encode(df_encoded, feature_name)
        elif self.encoding_type == 'label':
            df_encoded = self._label_encode(df_encoded, feature_name)
        elif self.encoding_type == 'ordinal':
            df_encoded = self._ordinal_encode(df_encoded, feature_name)
        else:
            raise ValueError(f"Unsupported encoding type: {self.encoding_type}")

        return df_encoded

    def _one_hot_encode(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Performs one-hot encoding on the feature."""
        if feature_name not in self.encoders:
            self.encoders[feature_name] = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore'
            )
            # Fit on non-null values
            non_null_mask = ~df[feature_name].isnull()
            if non_null_mask.any():
                self.encoders[feature_name].fit(df.loc[non_null_mask, [feature_name]])

        # Transform non-null values
        non_null_mask = ~df[feature_name].isnull()
        if non_null_mask.any():
            # Get feature names
            feature_names = [f"{feature_name}_{cat}" for cat in self.encoders[feature_name].categories_[0]]
            self.feature_names[feature_name] = feature_names

            # Transform and create new columns
            encoded_data = self.encoders[feature_name].transform(df.loc[non_null_mask, [feature_name]])
            
            # Create new columns with encoded data
            for i, feature_name_new in enumerate(feature_names):
                df.loc[non_null_mask, feature_name_new] = encoded_data[:, i]
                # Fill missing values with 0
                df[feature_name_new] = df[feature_name_new].fillna(0)

        # Drop original column
        df = df.drop(columns=[feature_name])
        return df

    def _label_encode(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Performs label encoding on the feature."""
        if feature_name not in self.encoders:
            self.encoders[feature_name] = LabelEncoder()
            # Fit on non-null values
            non_null_mask = ~df[feature_name].isnull()
            if non_null_mask.any():
                self.encoders[feature_name].fit(df.loc[non_null_mask, feature_name])

        # Transform non-null values
        non_null_mask = ~df[feature_name].isnull()
        if non_null_mask.any():
            df.loc[non_null_mask, feature_name] = self.encoders[feature_name].transform(
                df.loc[non_null_mask, feature_name]
            )
            # Fill missing values with -1
            df[feature_name] = df[feature_name].fillna(-1)

        return df

    def _ordinal_encode(self, df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
        """Performs ordinal encoding on the feature."""
        if feature_name not in self.encoders:
            self.encoders[feature_name] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Fit on non-null values
            non_null_mask = ~df[feature_name].isnull()
            if non_null_mask.any():
                self.encoders[feature_name].fit(df.loc[non_null_mask, [feature_name]])

        # Transform non-null values
        non_null_mask = ~df[feature_name].isnull()
        if non_null_mask.any():
            df.loc[non_null_mask, feature_name] = self.encoders[feature_name].transform(
                df.loc[non_null_mask, [feature_name]]
            )
            # Fill missing values with -1
            df[feature_name] = df[feature_name].fillna(-1)

        return df

    def get_feature_names(self, feature_name: str) -> List[str]:
        """Returns the names of encoded features for a given original feature."""
        if self.encoding_type == 'one_hot':
            return self.feature_names.get(feature_name, []) + [self.missing_indicators[feature_name]]
        else:
            return [feature_name, self.missing_indicators[feature_name]]


# Example usage
if __name__ == "__main__":
    # Example dataframe
    df = pd.read_csv('Dataset/Extracted/train.csv')
    
    # Print original data information
    print("\nOriginal DataFrame Info:")
    print(df[['installation_type']].head())
    print("\nMissing values in installation_type:")
    print(df['installation_type'].isnull().sum())
    
    # 1. One-Hot Encoding with missing value indicator
    print("\n=== One-Hot Encoding with Missing Value Indicator ===")
    encoder = CategoricalFeatureEncoder(encoding_type='one_hot')
    df_encoded = encoder.encode_feature(df, 'installation_type')
    
    print("\nEncoded DataFrame (first 5 rows):")
    print(df_encoded[encoder.get_feature_names('installation_type')].head())
    
    print("\nFeature names created:")
    print(encoder.get_feature_names('installation_type'))
    
    # 2. Label Encoding with missing value indicator
    print("\n=== Label Encoding with Missing Value Indicator ===")
    encoder = CategoricalFeatureEncoder(encoding_type='label')
    df_encoded = encoder.encode_feature(df, 'installation_type')
    
    print("\nEncoded DataFrame (first 5 rows):")
    print(df_encoded[encoder.get_feature_names('installation_type')].head())
    
    # 3. Ordinal Encoding with missing value indicator
    print("\n=== Ordinal Encoding with Missing Value Indicator ===")
    encoder = CategoricalFeatureEncoder(encoding_type='ordinal')
    df_encoded = encoder.encode_feature(df, 'installation_type')
    
    print("\nEncoded DataFrame (first 5 rows):")
    print(df_encoded[encoder.get_feature_names('installation_type')].head()) 