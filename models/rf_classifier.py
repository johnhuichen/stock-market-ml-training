import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier
import joblib

from models.model import Model

N_CORES = joblib.cpu_count(only_physical_cores=True)


class RFClassifierModel(Model):
    def __init__(
        self,
        n_estimators=100,
        min_samples_leaf=5,
        criterion="gini",
        n_jobs=N_CORES,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.n_jobs = n_jobs
        self.init_model()

    def __str__(self):
        return str(self.model)

    def init_model(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            n_jobs=self.n_jobs,
        )

    def fit(self, train_x: pandas.DataFrame, train_y: pandas.DataFrame) -> None:
        self.model.fit(train_x, train_y.values.ravel())

    def predict(self, val_x: pandas.DataFrame) -> numpy.ndarray:
        return self.model.predict(val_x)
