import os
from typing import Dict, List, Tuple

import boto3
import mlflow
import pandas as pd
from dotenv import load_dotenv

from src.config import Config
from src.data.data_manager import DataManager
from src.models.model_trainer import ModelTrainer
from src.models.ensemble_creator import EnsembleCreator
from src.evaluation.evaluator import Evaluator
from src.utils.s3_utils import S3Utils
from src.utils.mlflow_utils import MLflowUtils
from src.reporting.evidently_reporter import EvidentlyReporter

# Load environment variables
load_dotenv()

# Initialize configuration
config = Config()

# Initialize utilities
s3_utils = S3Utils(config)
mlflow_utils = MLflowUtils(config)

def run_experiment():
    with mlflow.start_run():
        # Data preparation
        data_manager = DataManager(config)
        X, y, cat_cols, num_cols = data_manager.load_and_preprocess_data()
        x_train, x_test, y_train, y_test = data_manager.split_data(X, y)

        # Generate reports
        evidently_reporter = EvidentlyReporter(config)
        evidently_reporter.generate_reports(x_train, y_train, x_test, y_test)

        # Log data info
        mlflow_utils.log_data_info(X, cat_cols, num_cols)

        # Train models
        model_trainer = ModelTrainer(config)
        models = model_trainer.train_models(x_train, y_train)

        # Create ensemble models
        ensemble_creator = EnsembleCreator(config)
        ensemble_models = ensemble_creator.create_ensemble_models(models, x_train, y_train)
        models.update(ensemble_models)

        # Evaluate models
        evaluator = Evaluator(config)
        results, best_model = evaluator.evaluate_models(models, x_train, y_train, x_test, y_test)

        # Save and upload best model
        if best_model:
            evaluator.save_best_model(best_model)
            s3_utils.upload_model_artifacts()

        # Create and upload performance plot
        evaluator.create_performance_plot(results)
        s3_utils.upload_performance_plot()

def main():
    print("Starting Heart Disease Classification project...")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    s3_utils.create_s3_bucket()
    
    run_experiment()
    
    print("Heart Disease Classification project completed.")

if __name__ == "__main__":
    main()
