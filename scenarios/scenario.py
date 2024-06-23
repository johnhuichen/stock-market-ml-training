import pandas

from models.model import Model
from metrics.metric import Metric


class Scenario:
    def train(self, model: Model) -> Metric:
        raise Exception("not implemented")

    def train_mean_metric(self, model: Model, epochs: int = 10) -> float:
        total_metric_value = 0.0
        for _ in range(epochs):
            model.init_model()
            total_metric_value += self.train(model).value()
        return total_metric_value / epochs

    def train_test_split(self) -> None:
        raise Exception("not implemented")

    def sorted_predictions(self) -> pandas.DataFrame:
        raise Exception("not implemented")
