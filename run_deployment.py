from Pipelines.DeploymentPipeline import deployment_pipeline
from config import SystemConfig

if __name__ == "__main__":
    config = SystemConfig()
    # Example model path (update as needed)
    model_path = "artifacts/model.joblib"
    deployment_pipeline(
        model_path=model_path,
        test_data_path=config.test_data_path,
        feature_columns=config.feature_columns,
        missing_strategy=config.missing_strategy,
        missing_columns=config.missing_columns
    )
    print("Deployment pipeline run complete.")
