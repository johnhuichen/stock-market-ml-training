import pandas as pd
from models.model import Model

from typing import Any


class TopPercent(Model):
    def __init__(self, top_percent: float, sort_by: str, ascending: bool = False):
        self.top_percent = top_percent
        self.sort_by = sort_by
        self.ascending = ascending

    def __str__(self) -> str:
        return f"Top Percent Model (top_percent={self.top_percent*100:.2f}%, sort_by={self.sort_by}, ascending={self.ascending})"

    def fit(self, x: Any, y: Any) -> None:
        pass

    def predict(self, x: pd.DataFrame, dataset: pd.DataFrame) -> Any:
        selected = dataset.sort_values(by=self.sort_by, ascending=self.ascending)
        selected = selected.head(int(len(selected) * self.top_percent))
        return x.isin(selected).iloc[:, [0]].astype(int).to_numpy()

    def visualize(self) -> None:
        pass
