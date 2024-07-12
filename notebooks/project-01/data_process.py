import os
from typing import List, Tuple

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
DATA_DIR = "./data"
MODEL_DIR = "./models"
MLFLOW_TRACKING_URI = "sqlite:///data/mlflow.db"
EXPERIMENT_NAME = "Heart Disease Data Processing"


def setup_directories():
    """Ensure necessary directories exist."""
    for directory in [DATA_DIR, MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)


def setup_mlflow():
    """Set up MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def read_data(data_path: str) -> pd.DataFrame:
    """Read data from CSV file."""
    df = pd.read_csv(data_path)
    mlflow.log_param("data_source", data_path)
    mlflow.log_metrics({"num_rows": len(df), "num_columns": len(df.columns)})
    return df


def summarize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of the dataset."""
    summary = pd.DataFrame(
        {
            "COLS": data.columns,
            "COUNT": data.count().values,
            "NULL": data.isna().sum().values,
            "NUNIQUE": data.nunique().values,
            "FREQ": data.mode().iloc[0].values,
            "MIN": data.describe().T["min"].values,
            "MAX": data.describe().T["max"].values,
            "MEAN": data.describe().T["mean"].values,
            "STD": data.describe().T["std"].values,
        }
    )
    summary.to_csv(f"{DATA_DIR}/data_summary.csv", index=False)
    mlflow.log_artifact(f"{DATA_DIR}/data_summary.csv")
    return summary


def split_data_types(summary: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split features into categorical and numerical."""
    cat_value = summary[summary["NUNIQUE"] < 5]["COLS"].tolist()
    num_value = summary[summary["NUNIQUE"] >= 5]["COLS"].tolist()
    mlflow.log_params(
        {
            "num_categorical_features": len(cat_value),
            "num_numerical_features": len(num_value),
        }
    )
    return cat_value, num_value


def preprocess_data(
    df: pd.DataFrame, target_column: str = "heart disease"
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Preprocess the data."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    summary = summarize_data(df)
    cat_value, num_value = split_data_types(summary)

    cat_value = [col for col in cat_value if col != target_column]

    for col in cat_value:
        X[col] = X[col].astype("category")

    X = pd.get_dummies(X)
    y = y.map({1: 0, 2: 1})

    mlflow.log_params(
        {
            "num_features_after_encoding": X.shape[1],
            "target_distribution": dict(y.value_counts(normalize=True)),
        }
    )

    return X, y, cat_value, num_value


def prepare_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2023
    )
    mlflow.log_params({"train_size": len(x_train), "test_size": len(x_test)})
    return x_train, x_test, y_train, y_test


def save_processed_data(
    X: pd.DataFrame,
    y: pd.Series,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Save processed data."""
    X.to_csv(f"{DATA_DIR}/preprocessed_features.csv", index=False)
    y.to_csv(f"{DATA_DIR}/preprocessed_target.csv", index=False)
    joblib.dump(
        (x_train, x_test, y_train, y_test), f"{DATA_DIR}/train_test_split.joblib"
    )
    mlflow.log_artifacts(DATA_DIR)


def main(df: pd.DataFrame, target_column: str = "heart disease") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Preprocess the data."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    summary = summarize_data(df)
    cat_value, num_value = split_data_types(summary)

    cat_value = [col for col in cat_value if col != target_column]

    for col in cat_value:
        X[col] = X[col].astype("category")

    X = pd.get_dummies(X)
    
    # Ensure y is binary and 1d
    y = (y == 2).astype(int).squeeze()  # Assuming 2 is the positive class, adjust if necessary

    mlflow.log_params({
        "num_features_after_encoding": X.shape[1],
        "target_distribution": dict(y.value_counts(normalize=True))
    })

    return X, y, cat_value, num_value


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed data for serving."""
    return joblib.load(f"{DATA_DIR}/train_test_split.joblib")


if __name__ == "__main__":
    data_path = f"{DATA_DIR}/dataset_heart.csv"
    x_train, x_test, y_train, y_test = main(data_path)
