from typing import Any, Optional


class Model:
    def __str__(self) -> str:
        raise Exception("not implemented")

    def fit(self, x: Any, y: Any) -> None:
        raise Exception("not implemented")

    # passing dataset allows model to cheat
    def predict(self, x: Any, dataset: Any) -> Any:
        raise Exception("not implemented")

    def visualize(self) -> None:
        raise Exception("not implemented")
