# ⚡ Solar Power System Analyzer - MLOps

A robust, modular, and production-ready platform for solar power system data analysis, machine learning, and prediction. Built with **ZenML**, **Streamlit**, **MLflow**, and a rich Python data science stack, this project enables end-to-end workflows from data ingestion and EDA to model training, deployment, and inference—all with experiment tracking and a user-friendly web interface.

---

## 🚀 Features

### 1. **End-to-End ML Pipelines with ZenML**
- **Training Pipeline:** Data ingestion, missing value handling, feature engineering, outlier detection, model training, evaluation, and model saving.
- **Deployment Pipeline:** Loads trained models, processes new data, and generates predictions for deployment scenarios.
- **Inference Pipeline:** Dynamically loads models and produces predictions on new data, supporting batch inference.

### 2. **Modular ZenML Steps**
- **Data Ingestion:** Supports CSV and ZIP files, with extensible ingestion logic.
- **Missing Value Handling:** Multiple strategies (drop, mean, median, mode, constant, KNN, CatBoost, categorical fill).
- **Feature Engineering:** Categorical encoding, new feature creation (e.g., power, area).
- **Outlier Detection:** Z-score and IQR-based methods.
- **Model Building:** Supports Linear Regression, Random Forest, XGBoost, CatBoost.
- **Model Evaluation:** Regression metrics with MLflow logging.
- **Model Saving/Loading:** Robust serialization and MLflow model registry integration.
- **Prediction & Saving:** Batch predictions and artifact logging.

### 3. **Interactive Streamlit Web App**
- **EDA Tab:** Upload CSVs and perform:
  - Missing value analysis (counts, heatmaps)
  - Univariate, bivariate, and multivariate analysis (histograms, boxplots, scatterplots, correlation heatmaps, pairplots)
- **Prediction Tab:** Upload data, run the full inference pipeline, and download predictions.
- **Home & About Tabs:** Project overview and author information.
- **Production-Grade Logging:** All user actions, errors, and pipeline events are logged for traceability.

### 4. **Advanced EDA Utilities**
- Modular EDA code in `Analysis/AnalyzeSrc/` for:
  - Univariate, bivariate, and multivariate analysis
  - Missing value visualization
  - Encoding strategies and data inspection

### 5. **Experiment Tracking with MLflow**
- All model metrics, artifacts, and predictions are logged and tracked for reproducibility and comparison.

### 6. **Centralized Configuration**
- All pipeline and model parameters are managed via a single `config.py` dataclass for easy customization.

### 7. **Extensible and Production-Ready**
- Modular codebase with clear separation of concerns.
- Logging in every process (steps, pipelines, app).
- Ready for cloud or on-prem deployment.

---

## 🛠️ Frameworks & Libraries

- **ZenML**: Orchestrates modular, reproducible ML pipelines.
- **Streamlit**: Interactive web UI for EDA and prediction.
- **MLflow**: Experiment tracking, model registry, and artifact logging.
- **scikit-learn, XGBoost, CatBoost, LightGBM**: Model training and evaluation.
- **pandas, numpy, matplotlib, seaborn, plotly**: Data manipulation and visualization.
- **joblib**: Model serialization.
- **colorama**: Terminal color support.
- **Other**: Jupyter, Pillow, python-dateutil, threadpoolctl, etc.

See `requirements.txt` for the full list.

---

## 📁 Project Structure

```
.
├── App/                  # Streamlit web app (EDA, prediction, UI)
│   ├── app.py
│   ├── eda.py
│   └── predict.py
├── Steps/                # ZenML pipeline steps (modular ML logic)
│   ├── DataIngestionStep.py
│   ├── FeatureEngineeringStep.py
│   ├── HandleMissingValueStep.py
│   ├── OutlierDetectionStep.py
│   ├── ModelBuildingStep.py
│   ├── ModelEvaluationStep.py
│   ├── ModelSaverStep.py
│   ├── ModelLoaderStep.py
│   ├── PredictionStep.py
│   ├── PredictionsSaverStep.py
│   ├── DynamicModelLoaderStep.py
│   └── SplitFeaturesTargetStep.py
├── Pipelines/            # ZenML pipeline orchestrations
│   ├── TrainingPipeline.py
│   ├── InferencePipeline.py
│   └── DeploymentPipeline.py
├── Src/                  # Core ML/data logic (feature engineering, ingestion, etc.)
├── Analysis/AnalyzeSrc/  # Advanced EDA utilities
├── config.py             # Centralized configuration (SystemConfig)
├── run_training.py       # Script to run the training pipeline
├── run_inference.py      # Script to run the inference pipeline
├── run_deployment.py     # Script to run the deployment pipeline
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### 1. **Install Requirements**
```bash
pip install -r requirements.txt
```

### 2. **Train a Model**
```bash
python run_training.py
```
- Uses parameters from `config.py`.
- Trains and saves a model to `artifacts/model.joblib`.

### 3. **Run Inference**
```bash
python run_inference.py --data_path path/to/input.csv --feature_columns col1,col2,...
```
- Produces predictions in `artifacts/predictions.csv`.

### 4. **Run Deployment Pipeline**
```bash
python run_deployment.py
```
- Loads a trained model and generates predictions on test data.

### 5. **Launch the Streamlit App**
```bash
cd App
streamlit run app.py
```
- Explore EDA and make predictions via the web UI.

---

## 📊 Streamlit App Features

- **EDA Tab:** Upload data, visualize missing values, distributions, relationships, and correlations.
- **Prediction Tab:** Upload new data, run the full inference pipeline, and download results.
- **Logging:** All actions and errors are logged to `App/app.log` for easy debugging.

---

## 🧩 Customization

- **Pipeline Parameters:** Edit `config.py` to change data paths, model types, feature columns, and more.
- **Add Steps:** Extend the `Steps/` directory with new ZenML steps for custom logic.
- **EDA:** Add or modify EDA modules in `Analysis/AnalyzeSrc/`.

---

## 📈 Experiment Tracking

- All model metrics, artifacts, and predictions are logged with **MLflow**.
- To view MLflow UI:
  ```bash
  mlflow ui
  ```
  Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 👀 Visualize and Manage Pipelines with ZenML Dashboard

ZenML provides a built-in dashboard to visualize, monitor, and manage your pipelines, steps, and artifacts.

To launch the ZenML dashboard, simply run:

```bash
zenml up
```

This will start the ZenML dashboard locally. Open your browser and go to [http://localhost:8237](http://localhost:8237) to:
- View all pipeline runs and their statuses
- Inspect step outputs, artifacts, and logs
- Monitor experiment lineage and metadata
- Manage stacks, orchestrators, and more

The dashboard is a powerful tool for tracking your ML workflow and debugging pipeline executions.

---

## 📝 Logging

- **App logs:** `App/app.log`
- **Pipeline logs:** `Pipelines/pipeline.log`
- **Step logs:** `Steps/step.log`
- All major actions, transitions, and errors are traceable for robust monitoring and debugging.

---

## 🏗️ Design Patterns & Best Practices

### Design Patterns Used
- **Strategy Pattern:** Used extensively for modularizing logic in data ingestion, missing value handling, feature engineering, outlier detection, model building, and EDA. This allows easy swapping and extension of algorithms and behaviors at runtime.
- **Factory Pattern:** Used for creating data ingestors based on file type, enabling scalable and maintainable data ingestion logic.
- **Modularization & Separation of Concerns:** The codebase is organized into clear modules (Steps, Pipelines, Src, Analysis) to ensure each component has a single responsibility and can be developed, tested, and maintained independently.

### Best Practices Followed
- **Production-Grade Logging:** All major actions, errors, and pipeline events are logged for traceability and debugging.
- **Centralized Configuration:** All parameters and settings are managed via a single config file for easy customization and reproducibility.
- **Experiment Tracking:** All model metrics, artifacts, and predictions are logged with MLflow for reproducibility and comparison.
- **Extensibility:** The use of abstract base classes and modular steps makes it easy to add new features or algorithms.
- **Reproducibility:** Pipelines, experiment tracking, and configuration management ensure results can be reliably reproduced.
- **Clear Documentation:** The project is well-documented with docstrings, comments, and this README.

---

## 🖼️ Images

Below are key images illustrating the system's architecture, pipelines, and results:

![deployment_pipeline](images/deployment_pipeline.png)
*Deployment Pipeline*

![EDA1](images/EDA1.png)
*Exploratory Data Analysis 1*

![EDA2](images/EDA2.png)
*Exploratory Data Analysis 2*

![EDA3](images/EDA3.png)
*Exploratory Data Analysis 3*

![inference_pipeline](images/inference_pipeline.png)
*Inference Pipeline*

![mlflow_experiment_tracking](images/mlflow_experiment_tracking.png)
*MLflow Experiment Tracking*

![predictions](images/predictions.png)
*Predictions Output*

![running_inference](images/running_inference.png)
*Running Inference*

![train_pipeline](images/train_pipeline.png)
*Training Pipeline*

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 👤 Author

**THAMIZHARASU SARAVANAN**  
[GitHub Profile](https://github.com/THAMIZH-ARASU)

---

## 📄 License

This project is licensed under [MIT License](LICENSE) 

---

**Enjoy robust, modular, and production-ready solar data analysis and ML!** 
