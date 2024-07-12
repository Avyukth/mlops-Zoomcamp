import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(experiment_name: str) -> str:
    """
    Set up MLflow tracking.
    
    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        str: Experiment ID
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    return experiment

def log_model_params(model):
    """
    Log model parameters to MLflow.
    
    Args:
        model: The trained model
    """
    params = model.get_params()
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

def log_model(model, model_name: str):
    """
    Log the model to MLflow.
    
    Args:
        model: The trained model
        model_name (str): Name of the model
    """
    mlflow.sklearn.log_model(model, model_name)
