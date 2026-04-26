import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass


# Concrete Strategy for Linear Regression using scikit-learn
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Linear Regression model with scaling.")

        # Creating a pipeline with standard scaling and linear regression
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),  # Feature scaling
                ("model", LinearRegression()),  # Linear regression model
            ]
        )

        logging.info("Training Linear Regression model.")
        pipeline.fit(X_train, y_train)  # Fit the pipeline to the training data

        logging.info("Model training completed.")
        return pipeline


# Concrete Strategy for Random Forest using scikit-learn
class RandomForestStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a Random Forest model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Random Forest model.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing Random Forest model with scaling.")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            ))
        ])

        logging.info("Training Random Forest model.")
        pipeline.fit(X_train, y_train)

        logging.info("Random Forest model training completed.")
        return pipeline


# Concrete Strategy for XGBoost
class XGBoostStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains an XGBoost model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained XGBoost model.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing XGBoost model with scaling.")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))
        ])

        logging.info("Training XGBoost model.")
        pipeline.fit(X_train, y_train)

        logging.info("XGBoost model training completed.")
        return pipeline


# Concrete Strategy for CatBoost
class CatBoostStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Builds and trains a CatBoost model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained CatBoost model.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Initializing CatBoost model with scaling.")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False
            ))
        ])

        logging.info("Training CatBoost model.")
        pipeline.fit(X_train, y_train)

        logging.info("CatBoost model training completed.")
        return pipeline


# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train)


# Example usage
if __name__ == "__main__":
    # Example DataFrame (replace with actual data loading)
    df = pd.read_csv('Dataset/Extracted/train.csv')
    X_train = df.drop(columns=['efficiency'])
    y_train = df['efficiency']

    # Example usage of different strategies
    model_builder = ModelBuilder(LinearRegressionStrategy())
    # trained_model = model_builder.build_model(X_train, y_train)
    
    # Switch to Random Forest
    model_builder.set_strategy(RandomForestStrategy())
    rf_model = model_builder.build_model(X_train, y_train)
    
    # Switch to XGBoost
    model_builder.set_strategy(XGBoostStrategy())
    xgb_model = model_builder.build_model(X_train, y_train)
    
    # Switch to CatBoost
    model_builder.set_strategy(CatBoostStrategy())
    catboost_model = model_builder.build_model(X_train, y_train)

    pass
