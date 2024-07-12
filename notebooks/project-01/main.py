import os
import mlflow
import pandas as pd
from typing import Dict, List, Tuple

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_splitter import DataSplitter
from src.utils.mlflow_utils import setup_mlflow
from src.models.base_model import BaseModel
from src.ensemble.voting import VotingEnsemble
from src.ensemble.stacking import StackingEnsemble
from src.evaluation.metrics import evaluate_model, log_feature_importance
from src.evaluation.visualizations import create_performance_plot, plot_feature_importance, plot_roc_curve
from src.utils.file_utils import load_config

# Constants
DATA_DIR = "./data/dataset_heart.csv"
MODEL_DIR = "./models"
CONFIG_PATH = "./config/model_params.yaml"
EXPERIMENT_NAME = "Heart Disease Classification"

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    preprocessor = DataPreprocessor(target_column="heart disease")
    X, y, cat_cols, num_cols = preprocessor.preprocess(df)
    
    return X, y, cat_cols, num_cols

def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(test_size=0.2, random_state=42)
    return splitter.split(X, y)

def train_models(x_train: pd.DataFrame, y_train: pd.Series, config: Dict) -> Dict[str, BaseModel]:
    from src.models.logistic_regression import LogisticRegressionModel
    from src.models.svm import SVMModel
    from src.models.knn import KNNModel
    from src.models.mlp import MLPModel
    from src.models.random_forest import RandomForestModel
    from src.models.extra_trees import ExtraTreesModel
    from src.models.catboost import CatBoostModel
    from src.models.lgbm import LGBMModel
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

def run_experiment(data_dir: str, config_path: str):
    mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Load and preprocess data
        X, y, cat_cols, num_cols = load_and_preprocess_data(data_dir)
        x_train, x_test, y_train, y_test = split_data(X, y)

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
        for name, model in models.items():
            train_score, test_score = evaluate_model(model, x_train, y_train, x_test, y_test)
            results[name] = {"train_score": train_score, "test_score": test_score}
            
            mlflow.log_metric(f"{name}_train_score", train_score)
            mlflow.log_metric(f"{name}_test_score", test_score)
            mlflow.sklearn.log_model(model, name)

            # Log feature importance if applicable
            log_feature_importance(model, X.columns)

        # Create and log performance plot
        create_performance_plot(results, f"{MODEL_DIR}/model_comparison.png")
def main():
    print("Starting Heart Disease Classification project...")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    run_experiment(DATA_DIR, CONFIG_PATH)
    
    print("Heart Disease Classification project completed.")

if __name__ == "__main__":
    main()
