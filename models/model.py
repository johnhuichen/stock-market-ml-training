from typing import Any


class Model:
    def __str__(self) -> str:
        raise Exception("not implemented")

    def init_model(self) -> None:
        raise Exception("not implemented")

    def fit(self, train_x: Any, train_y: Any) -> None:
        raise Exception("not implemented")

    def predict(self, val_x: Any) -> Any:
        raise Exception("not implemented")

    def visualize(self) -> None:
        raise Exception("not implemented")
