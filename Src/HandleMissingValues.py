import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Abstract method to handle missing values in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): List of column names to handle missing values for.
                                     If None, handles all columns.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        pass


# Concrete Strategy for Dropping Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """
        Initializes the DropMissingValuesStrategy with specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values.
        thresh (int): The threshold for non-NA values. Rows/Columns with less than thresh non-NA values are dropped.
        """
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Drops rows or columns with missing values based on the axis and threshold.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): List of column names to handle missing values for.
                                     If None, handles all columns.

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping missing values with axis={self.axis} and thresh={self.thresh}")
        if columns:
            df_cleaned = df.copy()
            df_cleaned[columns] = df[columns].dropna(axis=self.axis, thresh=self.thresh)
        else:
            df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


# Concrete Strategy for Filling Missing Values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Initializes the FillMissingValuesStrategy with a specific method or fill value.

        Parameters:
        method (str): The method to fill missing values ('mean', 'median', 'mode', or 'constant').
        fill_value (any): The constant value to fill missing values when method='constant'.
        """
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): List of column names to handle missing values for.
                                     If None, handles all columns.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        target_columns = columns if columns else df_cleaned.columns

        if self.method == "mean":
            numeric_columns = [col for col in target_columns if pd.api.types.is_numeric_dtype(df[col])]
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = [col for col in target_columns if pd.api.types.is_numeric_dtype(df[col])]
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in target_columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned[target_columns] = df_cleaned[target_columns].fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")

        logging.info("Missing values filled.")
        return df_cleaned


# Concrete Strategy for KNN-based Imputation
class KNNImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, n_neighbors=5, weights='uniform', handle_categorical=True):
        """
        Initializes the KNNImputationStrategy with specific parameters.

        Parameters:
        n_neighbors (int): Number of neighbors to use for imputation.
        weights (str): Weight function used in prediction ('uniform' or 'distance').
        handle_categorical (bool): Whether to handle categorical columns using label encoding.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.handle_categorical = handle_categorical
        self.imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        self.label_encoders = {}
        self.categorical_columns = []

    def _encode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Encodes categorical columns using LabelEncoder.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (List[str]): List of categorical columns to encode

        Returns:
        pd.DataFrame: DataFrame with encoded categorical columns
        """
        df_encoded = df.copy()
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on non-null values only
                non_null_mask = ~df[col].isnull()
                if non_null_mask.any():
                    self.label_encoders[col].fit(df.loc[non_null_mask, col])
            
            # Transform non-null values
            non_null_mask = ~df[col].isnull()
            if non_null_mask.any():
                df_encoded.loc[non_null_mask, col] = self.label_encoders[col].transform(
                    df.loc[non_null_mask, col]
                )
        
        return df_encoded

    def _decode_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Decodes encoded categorical columns back to original values.

        Parameters:
        df (pd.DataFrame): Input DataFrame with encoded values
        columns (List[str]): List of categorical columns to decode

        Returns:
        pd.DataFrame: DataFrame with decoded categorical columns
        """
        df_decoded = df.copy()
        for col in columns:
            if col in self.label_encoders:
                # Decode non-null values
                non_null_mask = ~df[col].isnull()
                if non_null_mask.any():
                    df_decoded.loc[non_null_mask, col] = self.label_encoders[col].inverse_transform(
                        df.loc[non_null_mask, col].astype(int)
                    )
        
        return df_decoded

    def handle(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Imputes missing values using K-Nearest Neighbors algorithm.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): List of column names to handle missing values for.
                                     If None, handles all columns.

        Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
        """
        logging.info(f"Imputing missing values using KNN with n_neighbors={self.n_neighbors}")
        
        df_cleaned = df.copy()
        
        # If no columns specified, use all columns
        if not columns:
            columns = df.columns.tolist()
        
        # Separate numeric and categorical columns
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_columns = [col for col in columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
        
        if not numeric_columns and not categorical_columns:
            logging.warning("No numeric or categorical columns found for KNN imputation")
            return df_cleaned

        # Handle categorical columns if enabled
        if self.handle_categorical and categorical_columns:
            logging.info(f"Encoding categorical columns: {categorical_columns}")
            df_cleaned = self._encode_categorical(df_cleaned, categorical_columns)
            self.categorical_columns = categorical_columns

        # Combine numeric and encoded categorical columns
        columns_to_impute = numeric_columns + (categorical_columns if self.handle_categorical else [])
        
        if columns_to_impute:
            # Perform KNN imputation
            imputed_data = self.imputer.fit_transform(df_cleaned[columns_to_impute])
            df_cleaned[columns_to_impute] = imputed_data

            # Decode categorical columns back to original values
            if self.handle_categorical and categorical_columns:
                logging.info("Decoding categorical columns back to original values")
                df_cleaned = self._decode_categorical(df_cleaned, categorical_columns)
        
        logging.info("KNN imputation completed.")
        return df_cleaned


# Concrete Strategy for Statistical Model-based Imputation
class StatisticalModelImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, model_type='linear', tune_hyperparameters=False, **model_params):
        """
        Initializes the StatisticalModelImputationStrategy with specific parameters.

        Parameters:
        model_type (str): Type of statistical model to use ('linear', 'random_forest', 'xgboost', 'lightgbm', or 'catboost').
        tune_hyperparameters (bool): Whether to perform hyperparameter tuning.
        model_params (dict): Additional parameters for the model.
        """
        self.model_type = model_type
        self.model_params = model_params
        self.tune_hyperparameters = tune_hyperparameters
        self.models = {}
        
        # Define default hyperparameter grids for tuning
        self.hyperparameter_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 63, 127]
            },
            'catboost': {
                'iterations': [50, 100, 200],
                'depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            }
        }

    def _get_model(self):
        """Returns the appropriate model based on model_type."""
        if self.model_type == 'linear':
            return LinearRegression(**self.model_params)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(**self.model_params)
        elif self.model_type == 'catboost':
            return CatBoostRegressor(**self.model_params, verbose=False)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _tune_hyperparameters(self, model, X, y):
        """Performs hyperparameter tuning for the given model."""
        if self.model_type not in self.hyperparameter_grids:
            return model

        grid_search = GridSearchCV(
            model,
            self.hyperparameter_grids[self.model_type],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        logging.info(f"Best parameters for {self.model_type}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def handle(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Imputes missing values using statistical models.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): List of column names to handle missing values for.
                                     If None, handles all numeric columns.

        Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
        """
        logging.info(f"Imputing missing values using {self.model_type} model")
        
        df_cleaned = df.copy()
        
        # If no columns specified, use all numeric columns
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter for numeric columns only
        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_columns:
            logging.warning("No numeric columns found for statistical model imputation")
            return df_cleaned

        # For each column with missing values, train a model using other columns as features
        for target_col in numeric_columns:
            if df[target_col].isnull().any():
                # Get rows where target column is not null for training
                train_mask = ~df[target_col].isnull()
                test_mask = df[target_col].isnull()
                
                if train_mask.sum() < 2:  # Need at least 2 samples for training
                    logging.warning(f"Not enough samples to train model for column {target_col}")
                    continue
                
                # Use other numeric columns as features
                feature_cols = [col for col in numeric_columns if col != target_col]
                if not feature_cols:
                    logging.warning(f"No features available for column {target_col}")
                    continue
                
                # Prepare training data
                X_train = df.loc[train_mask, feature_cols]
                y_train = df.loc[train_mask, target_col]
                X_test = df.loc[test_mask, feature_cols]
                
                # Get base model
                model = self._get_model()
                
                # Perform hyperparameter tuning if requested
                if self.tune_hyperparameters:
                    model = self._tune_hyperparameters(model, X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Predict missing values
                predictions = model.predict(X_test)
                df_cleaned.loc[test_mask, target_col] = predictions
                
                # Store the model for potential future use
                self.models[target_col] = model
                
                logging.info(f"Imputed missing values for column {target_col}")
        
        logging.info("Statistical model imputation completed.")
        return df_cleaned


# Concrete Strategy for Categorical Missing Value Feature
class CategoricalMissingFeatureStrategy(MissingValueHandlingStrategy):
    def __init__(self, column_name: str, fill_value: str):
        """
        Initializes the CategoricalMissingFeatureStrategy with specific parameters.

        Parameters:
        column_name (str): Name of the categorical column to handle.
        fill_value (str): Value to fill missing values with.
        """
        self.column_name = column_name
        self.fill_value = fill_value
        self.original_missing_mask = None

    def handle(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handles missing values in categorical columns by treating them as a feature.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): Not used in this strategy as we work with a specific column.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled and a new feature indicating missingness.
        """
        logging.info(f"Handling missing values in categorical column '{self.column_name}' as a feature")
        
        if self.column_name not in df.columns:
            raise ValueError(f"Column '{self.column_name}' not found in DataFrame")
        
        df_cleaned = df.copy()
        
        # Store the original missing mask
        self.original_missing_mask = df_cleaned[self.column_name].isnull()
        
        # Create a new column indicating missingness
        missing_feature_name = f"{self.column_name}_is_missing"
        df_cleaned[missing_feature_name] = self.original_missing_mask.astype(int)
        
        # Fill missing values with the specified value
        df_cleaned[self.column_name] = df_cleaned[self.column_name].fillna(self.fill_value)
        
        logging.info(f"Created missing indicator feature '{missing_feature_name}'")
        logging.info(f"Filled missing values in '{self.column_name}' with '{self.fill_value}'")
        
        return df_cleaned

    def get_missing_indicator(self) -> pd.Series:
        """
        Returns the missing indicator for the handled column.

        Returns:
        pd.Series: Boolean series indicating which values were originally missing.
        """
        if self.original_missing_mask is None:
            raise ValueError("Strategy has not been applied to any DataFrame yet")
        return self.original_missing_mask


# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.
        columns (List[str], optional): List of column names to handle missing values for.
                                     If None, handles all columns.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df, columns)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    df = pd.read_csv('Dataset/Extracted/train.csv')
    columns_to_clean = ['temperature', 'maintenance_count']

    # Dictionary to store results from different methods
    imputation_results = {}
    
    # 1. KNN Imputation
    missing_value_handler = MissingValueHandler(KNNImputationStrategy(n_neighbors=5))
    df_knn_imputed = missing_value_handler.handle_missing_values(df, columns=columns_to_clean)
    imputation_results['KNN'] = df_knn_imputed[columns_to_clean]
    print('KNN Imputed DataFrame:')
    print(df_knn_imputed[columns_to_clean].describe())

    # 2. XGBoost Imputation with hyperparameter tuning
    missing_value_handler.set_strategy(StatisticalModelImputationStrategy(
        model_type='xgboost',
        tune_hyperparameters=True
    ))
    df_xgb_imputed = missing_value_handler.handle_missing_values(df, columns=columns_to_clean)
    imputation_results['XGBoost'] = df_xgb_imputed[columns_to_clean]
    print('\nXGBoost Imputed DataFrame:')
    print(df_xgb_imputed[columns_to_clean].describe())

    # 3. LightGBM Imputation
    missing_value_handler.set_strategy(StatisticalModelImputationStrategy(
        model_type='lightgbm',
        n_estimators=100,
        learning_rate=0.1
    ))
    df_lgb_imputed = missing_value_handler.handle_missing_values(df, columns=columns_to_clean)
    imputation_results['LightGBM'] = df_lgb_imputed[columns_to_clean]
    print('\nLightGBM Imputed DataFrame:')
    print(df_lgb_imputed[columns_to_clean].describe())

    # 4. CatBoost Imputation
    missing_value_handler.set_strategy(StatisticalModelImputationStrategy(
        model_type='catboost',
        iterations=100,
        learning_rate=0.1
    ))
    df_cat_imputed = missing_value_handler.handle_missing_values(df, columns=columns_to_clean)
    imputation_results['CatBoost'] = df_cat_imputed[columns_to_clean]
    print('\nCatBoost Imputed DataFrame:')
    print(df_cat_imputed[columns_to_clean].describe())

    # 5. Categorical Missing Feature Strategy
    categorical_column = 'equipment_type'  # Replace with your categorical column name
    missing_value_handler.set_strategy(CategoricalMissingFeatureStrategy(
        column_name=categorical_column,
        fill_value='unknown'
    ))
    df_categorical_handled = missing_value_handler.handle_missing_values(df)
    
    # Print information about the categorical handling
    print('\nCategorical Missing Feature Handling:')
    print(f"Original missing values in {categorical_column}: {df[categorical_column].isnull().sum()}")
    print(f"Created missing indicator feature: {categorical_column}_is_missing")
    print("\nValue counts after handling:")
    print(df_categorical_handled[categorical_column].value_counts())
    print("\nMissing indicator value counts:")
    print(df_categorical_handled[f"{categorical_column}_is_missing"].value_counts())

    # Compare imputation results
    print("\n=== Comparison of Imputation Methods ===")
    
    # 1. Statistical Comparison
    print("\nStatistical Comparison:")
    comparison_stats = pd.DataFrame()
    
    for method, result_df in imputation_results.items():
        stats_dict = {}
        for col in columns_to_clean:
            stats_dict[f'{col}_mean'] = result_df[col].mean()
            stats_dict[f'{col}_std'] = result_df[col].std()
            stats_dict[f'{col}_min'] = result_df[col].min()
            stats_dict[f'{col}_max'] = result_df[col].max()
        comparison_stats[method] = pd.Series(stats_dict)
    
    print("\nComparison Statistics:")
    print(comparison_stats)

    # 2. Pairwise Differences
    print("\nPairwise Differences (Mean Absolute Difference):")
    methods = list(imputation_results.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            diff = np.abs(imputation_results[method1] - imputation_results[method2])
            print(f"\n{method1} vs {method2}:")
            print(diff.mean())

    # 3. Visualize Results
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each column
    for idx, col in enumerate(columns_to_clean, 1):
        plt.subplot(len(columns_to_clean), 1, idx)
        
        # Plot box plots for each method
        data_to_plot = [imputation_results[method][col] for method in methods]
        plt.boxplot(data_to_plot, labels=methods)
        plt.title(f'Distribution of Imputed Values - {col}')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('imputation_comparison.png')
    plt.close()

    # 4. Correlation Analysis
    print("\nCorrelation between different imputation methods:")
    for col in columns_to_clean:
        print(f"\nCorrelations for {col}:")
        corr_matrix = pd.DataFrame({
            method: imputation_results[method][col] 
            for method in methods
        }).corr()
        print(corr_matrix)

    # 5. Missing Value Analysis
    print("\nMissing Value Analysis:")
    for col in columns_to_clean:
        original_missing = df[col].isnull().sum()
        print(f"\n{col}:")
        print(f"Original missing values: {original_missing}")
        for method in methods:
            imputed_missing = imputation_results[method][col].isnull().sum()
            print(f"{method} remaining missing values: {imputed_missing}")

    # 6. Distribution Comparison
    print("\nDistribution Comparison (KS Test):")
    for col in columns_to_clean:
        print(f"\n{col}:")
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                ks_stat, p_value = stats.ks_2samp(
                    imputation_results[method1][col],
                    imputation_results[method2][col]
                )
                print(f"{method1} vs {method2}:")
                print(f"KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

    # 7. Save Results
    results_df = pd.DataFrame()
    for method, result_df in imputation_results.items():
        for col in columns_to_clean:
            results_df[f'{method}_{col}'] = result_df[col]
    
    results_df.to_csv('imputation_results.csv', index=False)
    print("\nResults have been saved to 'imputation_results.csv'")
