from Pipelines.TrainingPipeline import training_pipeline
from config import SystemConfig

if __name__ == "__main__":
    config = SystemConfig()
    training_pipeline(
        file_path=config.train_data_path,
        missing_strategy=config.missing_strategy,
        missing_columns=config.missing_columns,
        outlier_method=config.outlier_method,
        outlier_handle=config.outlier_handle,
        model_type=config.model_type,
        target_column=config.target_column,
        feature_columns=config.feature_columns
    )
