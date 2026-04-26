from zenml import pipeline
from Steps.PredictionStep import prediction_step
from Steps.DataIngestionStep import data_ingestion_step
from Steps.SplitFeaturesTargetStep import split_features_target_step
from Steps.DynamicModelLoaderStep import dynamic_model_loader_step
from Steps.FeatureEngineeringStep import feature_engineering_step
from Steps.PredictionsSaverStep import predictions_saver_step
import logging

# Configure logging
logging.basicConfig(
    filename="Pipelines/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@pipeline(enable_cache=False)
def inference_pipeline(
    model_path: str,
    data_path: str,
    feature_columns: list,
    model_import_path: str,
    missing_strategy: str = 'catboost',
    missing_columns: list = [],
):
    try:
        logging.info("InferencePipeline: Pipeline started.")
        model = dynamic_model_loader_step(model_path=model_path, import_path=model_import_path)
        logging.info("InferencePipeline: Model loader step completed.")
        df = data_ingestion_step(file_path=data_path)
        logging.info("InferencePipeline: Data ingestion step completed.")
        df_features = feature_engineering_step(df)
        logging.info("InferencePipeline: Feature engineering step completed.")
        X, _ = split_features_target_step(
            df=df_features,
            target_column=feature_columns[0],
            feature_columns=feature_columns
        )
        logging.info("InferencePipeline: Split features step completed.")
        predictions = prediction_step(model=model, X_test=X)
        logging.info("InferencePipeline: Prediction step completed.")
        predictions_saver_step(predictions=predictions, save_path="artifacts/predictions.csv")
        logging.info("InferencePipeline: Predictions saver step completed.")
        logging.info("InferencePipeline: Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"InferencePipeline: Error: {e}")
        raise
    print('-'*100)
    print(predictions) 
    print('-'*100)