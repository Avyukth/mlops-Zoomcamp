from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_splitter import DataSplitter
from src.models.logistic_regression import LogisticRegressionModel
from src.evaluation.metrics import evaluate_model
from src.utils.mlflow_utils import setup_mlflow

def main():
    # Setup MLflow
    experiment_id = setup_mlflow("Heart Disease Classification")

    # Load and preprocess data
    data_loader = DataLoader("data/dataset_heart.csv")
    df = data_loader.load_data()

    preprocessor = DataPreprocessor()
    X, y, cat_value, num_value = preprocessor.preprocess(df)

    # Split data
    data_splitter = DataSplitter()
    X_train, X_test, y_train, y_test = data_splitter.split(X, y)

    # Train and evaluate model
    model = LogisticRegressionModel(solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)

    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
