import pandas as pd
from models.model import Model

from typing import Any, Optional


class RandomSelect(Model):
    def __init__(self, frac: float):
        self.frac = frac

    def __str__(self) -> str:
        return f"Random Select Model (frac={self.frac*100:.2f}%)"

    def fit(self, x: Any, y: Any) -> None:
        pass

    def predict(self, x: pd.DataFrame, dataset: Any) -> Any:
        selected = x.sample(frac=self.frac)
        return x.isin(selected).iloc[:, [0]].astype(int).to_numpy()

    def visualize(self) -> None:
        pass
