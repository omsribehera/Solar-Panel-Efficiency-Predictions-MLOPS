from zenml import pipeline
from Steps.ModelLoaderStep import model_loader_step
from Steps.FeatureEngineeringStep import feature_engineering_step
from Steps.PredictionStep import prediction_step
from Steps.DataIngestionStep import data_ingestion_step
from Steps.HandleMissingValueStep import handle_missing_value_step
from Steps.SplitFeaturesTargetStep import split_features_target_step
import logging

# Configure logging
logging.basicConfig(
    filename="Pipelines/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@pipeline(enable_cache=False)
def deployment_pipeline(
    model_path: str,
    test_data_path: str,
    feature_columns: list,
    missing_strategy: str = 'catboost',
    missing_columns: list = [],
):
    try:
        logging.info("DeploymentPipeline: Pipeline started.")
        model = model_loader_step(model_path=model_path)
        logging.info("DeploymentPipeline: Model loader step completed.")
        df = data_ingestion_step(file_path=test_data_path)
        logging.info("DeploymentPipeline: Data ingestion step completed.")
        cols = missing_columns if missing_columns else None
        df_clean = handle_missing_value_step(df=df, strategy=missing_strategy, columns=cols)
        logging.info("DeploymentPipeline: Handle missing value step completed.")
        df_features = feature_engineering_step(df_clean)
        logging.info("DeploymentPipeline: Feature engineering step completed.")
        X_test, _ = split_features_target_step(
            df=df_features,
            target_column=feature_columns[0],
            feature_columns=feature_columns
        )
        logging.info("DeploymentPipeline: Split features step completed.")
        predictions = prediction_step(model=model, X_test=X_test)
        logging.info("DeploymentPipeline: Prediction step completed.")
        logging.info("DeploymentPipeline: Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"DeploymentPipeline: Error: {e}")
        raise
