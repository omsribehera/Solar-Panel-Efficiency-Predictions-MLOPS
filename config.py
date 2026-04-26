from dataclasses import dataclass, field
from typing import List

@dataclass
class SystemConfig:
    # Data paths
    train_data_path: str = "Dataset/Extracted/train.csv"
    test_data_path: str = "Dataset/Extracted/test.csv"
    sample_submission_path: str = "Dataset/Extracted/sample_submission.csv"

    # Missing value handling
    missing_strategy: str = "catboost"
    missing_columns: List[str] = field(default_factory=lambda: [
        'temperature', 'irradiance', 'panel_age', 'maintenance_count',
        'soiling_ratio', 'voltage', 'current', 'module_temperature', 'cloud_coverage'
    ])
    fill_value: str = "No Error"

    # Outlier detection
    outlier_method: str = "iqr"
    outlier_handle: str = "remove"

    # Model parameters
    model_type: str = "random_forest"
    target_column: str = "efficiency"
    feature_columns: List[str] = field(default_factory=lambda: [
        'temperature', 'irradiance', 'module_temperature', 'power'
    ])

    # Other pipeline options can be added here 