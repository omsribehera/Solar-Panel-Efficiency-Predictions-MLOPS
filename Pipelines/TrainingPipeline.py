from zenml import pipeline
from Steps.DataIngestionStep import data_ingestion_step
from Steps.HandleMissingValueStep import handle_missing_value_step
from Steps.FeatureEngineeringStep import feature_engineering_step
from Steps.OutlierDetectionStep import outlier_detection_step
from Steps.ModelBuildingStep import model_building_step
from Steps.PredictionStep import prediction_step
from Steps.SplitFeaturesTargetStep import split_features_target_step
from Steps.ModelEvaluationStep import model_evaluation_step
from Steps.ModelSaverStep import model_saver_step
import logging

# Configure logging
logging.basicConfig(
    filename="Pipelines/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@pipeline(enable_cache=False)
def training_pipeline(
    file_path: str,
    missing_strategy: str = 'catboost',
    missing_columns: list = [],
    outlier_method: str = 'iqr',
    outlier_handle: str = 'remove',
    model_type: str = 'random_forest',
    target_column: str = 'area',
    feature_columns: list = []
):
    try:
        logging.info("TrainingPipeline: Pipeline started.")
        df = data_ingestion_step(file_path)
        logging.info("TrainingPipeline: Data ingestion step completed.")
        cols = missing_columns if missing_columns else None
        df_clean = handle_missing_value_step(df=df, strategy=missing_strategy, columns=cols)
        logging.info("TrainingPipeline: Handle missing value step completed.")
        df_features = feature_engineering_step(df_clean)
        logging.info("TrainingPipeline: Feature engineering step completed.")
        df_no_outliers = outlier_detection_step(df=df_features, method=outlier_method, handle=outlier_handle)
        logging.info("TrainingPipeline: Outlier detection step completed.")
        X_train, y_train = split_features_target_step(
            df=df_no_outliers,
            target_column=target_column,
            feature_columns=feature_columns if feature_columns else [col for col in df_no_outliers.columns if col != target_column]
        )
        logging.info("TrainingPipeline: Split features/target step completed.")
        model = model_building_step(X_train=X_train, y_train=y_train, model_type=model_type)
        logging.info("TrainingPipeline: Model building step completed.")
        predictions = prediction_step(model=model, X_test=X_train)
        logging.info("TrainingPipeline: Prediction step completed.")
        metrics = model_evaluation_step(model=model, X_test=X_train, y_test=y_train)
        logging.info(f"TrainingPipeline: Model evaluation step completed. Metrics: {metrics}")
        model_saver_step(model=model, save_path="artifacts/model.joblib")
        logging.info("TrainingPipeline: Model saver step completed.")
        logging.info("TrainingPipeline: Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"TrainingPipeline: Error: {e}")
        raise