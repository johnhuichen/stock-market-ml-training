import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import re

from models.model import Model

from typing import Any


class DecisionTree(Model):
    def __init__(self, max_leaf_nodes=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.model = DecisionTreeClassifier(max_leaf_nodes=5)
        self.columns = []

    def __str__(self):
        return f"Decision Tree Model (max_leaf_nodes={self.max_leaf_nodes})"

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.columns = x.columns
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame, dataset: Any) -> np.ndarray:
        return self.model.predict(x)

    def visualize(self) -> None:
        size = 10
        ratio = 0.6
        precision = 7

        s = export_graphviz(
            self.model,
            out_file=None,
            feature_names=self.columns,
            filled=True,
            rounded=True,
            special_characters=True,
            rotate=False,
            precision=precision,
        )
        graphviz.Source(
            re.sub("Tree {", f"Tree {{ size={size}; ratio={ratio}", s)
        ).view()
