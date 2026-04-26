from Pipelines.InferencePipeline import inference_pipeline
from config import SystemConfig
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the inference pipeline.")
    parser.add_argument("--model_path", type=str, default="artifacts/model.joblib", help="Path to the trained model.")
    parser.add_argument("--data_path", type=str, help="Path to the input data CSV.")
    parser.add_argument("--feature_columns", type=str, help="Comma-separated list of feature columns.")
    parser.add_argument("--model_import_path", type=str, default="joblib:load", help="Import path for the model loader.")
    args = parser.parse_args()

    config = SystemConfig()

    # Use CLI args if provided, otherwise fall back to config.py
    data_path = args.data_path if args.data_path else getattr(config, 'test_data_path', None)
    feature_columns = (
        [col.strip() for col in args.feature_columns.split(",")] if args.feature_columns
        else getattr(config, 'feature_columns', None)
    )

    if not data_path or not feature_columns:
        raise ValueError("You must provide --data_path and --feature_columns as arguments or set them in config.py.")

    inference_pipeline(
        model_path=args.model_path,
        data_path=data_path,
        feature_columns=feature_columns,
        model_import_path=args.model_import_path
    )
    print("Inference pipeline run complete.") 