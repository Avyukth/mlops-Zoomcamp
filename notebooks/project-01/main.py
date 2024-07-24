import os
from typing import Dict, List, Tuple

import boto3
import mlflow
import pandas as pd
from dotenv import load_dotenv
from evidently.metric_preset import (DataDriftPreset, DataQualityPreset,
                                     TargetDriftPreset)
from evidently.report import Report
from evidently.test_preset import (DataQualityTestPreset,
                                   DataStabilityTestPreset)
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_splitter import DataSplitter
from src.ensemble.stacking import StackingEnsemble
from src.ensemble.voting import VotingEnsemble
from src.evaluation.metrics import evaluate_model, log_feature_importance
from src.evaluation.visualizations import (create_performance_plot,
                                           plot_feature_importance,
                                           plot_roc_curve)
from src.models.base_model import BaseModel
from src.utils.file_utils import load_config, load_model, save_model
from src.utils.mlflow_utils import setup_mlflow

# Constants
DATA_DIR = "./data/dataset_heart.csv"
MODEL_DIR = "./models"
CONFIG_PATH = "./config/model_params.yaml"
EXPERIMENT_NAME = "Heart Disease Classification"
S3_BUCKET_NAME = "heart-disease-models"


load_dotenv()

endpoint_url = os.getenv('AWS_ENDPOINT_URL')
access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION_NAME')


s3_client = boto3.client(
    's3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    region_name=region_name
)


def upload_to_s3(file_path: str, object_name: str):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, object_name)
        print(f"Uploaded {file_path} to S3 bucket {S3_BUCKET_NAME}")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")



def load_and_preprocess_data(data_path: str, model_dir: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    # Load data
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor(target_column="heart disease")
    X, y = preprocessor.fit_transform(df)
    
    # Get categorical and numerical column information
    cat_cols = preprocessor.feature_preprocessor.cat_cols
    num_cols = preprocessor.feature_preprocessor.num_cols
    
    # Log preprocessing info
    preprocessor.log_preprocessing_info(X, y)
    
    # Save preprocessors
    feature_preprocessor_path = os.path.join(model_dir, "feature_preprocessor.joblib")
    target_preprocessor_path = os.path.join(model_dir, "target_preprocessor.joblib")
    preprocessor.save(feature_preprocessor_path, target_preprocessor_path)
    
    # Log preprocessors as artifacts
    mlflow.log_artifact(feature_preprocessor_path, "preprocessors")
    mlflow.log_artifact(target_preprocessor_path, "preprocessors")

    return X, y, cat_cols, num_cols

def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(test_size=0.2, random_state=42)
    return splitter.split(X, y)

def train_models(x_train: pd.DataFrame, y_train: pd.Series, config: Dict) -> Dict[str, BaseModel]:
    from src.models.catboost import CatBoostModel
    from src.models.extra_trees import ExtraTreesModel
    from src.models.knn import KNNModel
    from src.models.lgbm import LGBMModel
    from src.models.logistic_regression import LogisticRegressionModel
    from src.models.mlp import MLPModel
    from src.models.random_forest import RandomForestModel
    from src.models.svm import SVMModel
    from src.models.xgboost import XGBoostModel

    models = {
        "Logistic": LogisticRegressionModel(**config['logistic_regression']),
        "SVM": SVMModel(**config['svm']),
        "KNN": KNNModel(**config['knn']),
        "MLP": MLPModel(**config['mlp']),
        "RandomForest": RandomForestModel(**config['random_forest']),
        "ExtraTrees": ExtraTreesModel(**config['extra_trees']),
        "CatBoost": CatBoostModel(**config['catboost']),
        "LGBM": LGBMModel(**config['lgbm']),
        "XGB": XGBoostModel(**config['xgboost'])
    }

    for name, model in models.items():
        model.fit(x_train, y_train)

    return models

def create_ensemble_models(models: Dict[str, BaseModel], x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, BaseModel]:
    voting_soft = VotingEnsemble(list(models.items()), voting='soft')
    voting_hard = VotingEnsemble(list(models.items()), voting='hard')
    stacking_logistic = StackingEnsemble(list(models.values()), meta_model=models['Logistic'])
    stacking_lgbm = StackingEnsemble(list(models.values()), meta_model=models['LGBM'])

    ensemble_models = {
        "Ensemble_Soft": voting_soft,
        "Ensemble_Hard": voting_hard,
        "Stacking_Logistic": stacking_logistic,
        "Stacking_LGBM": stacking_lgbm
    }

    # Fit the ensemble models
    for name, model in ensemble_models.items():
        print(f"Fitting {name}...")
        model.fit(x_train, y_train)

    return ensemble_models

def run_experiment(data_dir: str, model_dir: str, config_path: str):
    mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Load and preprocess data
        X, y, cat_cols, num_cols = load_and_preprocess_data(data_dir, model_dir)
        x_train, x_test, y_train, y_test = split_data(X, y)

        generate_evidently_reports(x_train, y_train, x_test, y_test)

        # Log data info
        mlflow.log_param("data_path", data_dir)
        mlflow.log_param("num_samples", len(X))
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("num_categorical_features", len(cat_cols))
        mlflow.log_param("num_numerical_features", len(num_cols))

        # Load config and train models
        config = load_config(config_path)
        models = train_models(x_train, y_train, config)

        # Create and fit ensemble models
        ensemble_models = create_ensemble_models(models, x_train, y_train)
        models.update(ensemble_models)

        # Evaluate and log results
        results = {}
        best_score = 0
        for name, model in models.items():
            train_score, test_score = evaluate_model(model, x_train, y_train, x_test, y_test)
            results[name] = {"train_score": train_score, "test_score": test_score}
            
            mlflow.log_metric(f"{name}_train_score", train_score)
            mlflow.log_metric(f"{name}_test_score", test_score)
            mlflow.sklearn.log_model(model, name)

            # Log feature importance if applicable
            log_feature_importance(model, X.columns)
                        # Keep track of the best model
            if test_score > best_score:
                best_score = test_score
                best_model = model

        # Save the best model
        if best_model:
            best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
            save_model(best_model, "best_model", MODEL_DIR)
            mlflow.log_artifact(best_model_path, "best_model")
            
            # Upload best model to S3
            upload_to_s3(best_model_path, "best_model.joblib")
            
            preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
            mlflow.log_artifact(preprocessor_path, "preprocessor")
            
            # Upload preprocessor to S3
            upload_to_s3(preprocessor_path, "preprocessor.joblib")

        # Create and log performance plot
        performance_plot_path = f"{MODEL_DIR}/model_comparison.png"
        create_performance_plot(results, performance_plot_path)
        
        # Upload performance plot to S3
        upload_to_s3(performance_plot_path, "model_comparison.png")



def generate_evidently_reports(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
    """Generate Evidently Test Suite and Report."""
    # Combine features and target for each dataset
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    # Create an Evidently Report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset()
    ])

    report.run(reference_data=train_data, current_data=test_data)
    report_path = os.path.join(MODEL_DIR, "evidently_report.html")
    report.save_html(report_path)
    mlflow.log_artifact(report_path, "evidently_reports")

    # Create an Evidently Test Suite
    test_suite = TestSuite(tests=[
        DataStabilityTestPreset(),
        DataQualityTestPreset(),
        TestColumnDrift(column_name="age"),
        TestColumnDrift(column_name="sex"),
        TestColumnDrift(column_name="chest_pain_type"),
        TestColumnDrift(column_name="resting_blood_pressure"),
    ])

    test_suite.run(reference_data=train_data, current_data=test_data)
    test_suite_path = os.path.join(MODEL_DIR, "evidently_test_suite.html")
    test_suite.save_html(test_suite_path)
    mlflow.log_artifact(test_suite_path, "evidently_reports")

    print(f"Evidently reports saved to {report_path} and {test_suite_path}")
def main():
    print("Starting Heart Disease Classification project...")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
        print(f"Created S3 bucket: {S3_BUCKET_NAME}")
    except s3_client.exceptions.BucketAlreadyExists:
        print(f"S3 bucket already exists: {S3_BUCKET_NAME}")
    except Exception as e:
        print(f"Error creating S3 bucket: {str(e)}")
    
    run_experiment(DATA_DIR, MODEL_DIR, CONFIG_PATH)
    
    print("Heart Disease Classification project completed.")

if __name__ == "__main__":
    main()
