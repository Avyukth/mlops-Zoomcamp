from typing import List, Tuple
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from src.models.base_model import BaseModel

class VotingEnsemble(BaseModel):
    def __init__(self, models: List[Tuple[str, BaseModel]], voting: str = 'soft'):
        self.models = models
        self.voting = voting
        self.ensemble = VotingClassifier(
            estimators=[(name, model.model) for name, model in models],
            voting=voting
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.ensemble.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.ensemble.predict(X)

    def get_params(self) -> dict:
        return self.ensemble.get_params()
