from models.model import Model
from trainers.metric import Metric


class Trainer:
    def train(self, model: Model) -> Metric:
        raise Exception("not implemented")

    def shuffle_train_test(self) -> None:
        raise Exception("not implemented")
