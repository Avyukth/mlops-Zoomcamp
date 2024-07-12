from .base_model import BaseModel
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class SVMModel(BaseModel):
    def __init__(self, **kwargs):
        self.model = Pipeline([
            ('MinMaxScale', MinMaxScaler()),
            ('SVC', SVC(probability=True, **kwargs))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.model.named_steps['SVC'].get_params()
