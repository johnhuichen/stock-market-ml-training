import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import re

from models.model import Model


class DecisionTree(Model):
    def __init__(self, max_leaf_nodes=5):
        self.max_leaf_nodes = max_leaf_nodes
        self.model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        self.columns = []

    def __str__(self):
        return f"Decision Tree Model (max_leaf_nodes={self.max_leaf_nodes})"

    def fit(self, train_x: pandas.DataFrame, train_y: pandas.DataFrame) -> None:
        self.columns = train_x.columns
        self.model.fit(train_x, train_y)

    def predict(self, val_x: pandas.DataFrame) -> numpy.ndarray:
        return self.model.predict(val_x)

    def visualize(self) -> None:
        size = 10
        ratio = 0.6
        precision = 7

        s = str(
            export_graphviz(
                self.model,
                out_file=None,
                feature_names=self.columns,
                filled=True,
                rounded=True,
                special_characters=True,
                rotate=False,
                precision=precision,
            )
        )
        graphviz.Source(
            re.sub("Tree {", f"Tree {{ size={size}; ratio={ratio}", s)
        ).view()
