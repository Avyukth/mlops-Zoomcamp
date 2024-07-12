from typing import List
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from src.models.base_model import BaseModel

class StackingEnsemble(BaseModel):
    def __init__(self, models: List[BaseModel], meta_classifier=None):
        if meta_classifier is None:
            meta_classifier = LogisticRegression()
        
        self.models = models
        self.ensemble = StackingClassifier(
            estimators=[(f"model_{i}", model.model) for i, model in enumerate(models)],
            final_estimator=meta_classifier
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.ensemble.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.ensemble.predict(X)

    def get_params(self) -> dict:
        return self.ensemble.get_params()
