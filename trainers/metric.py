import functools


@functools.total_ordering
class Metric:
    def __str__(self) -> str:
        raise Exception("not implemented")

    def __eq__(self, other) -> bool:
        raise Exception("not implemented")

    def __lt__(self, other) -> bool:
        raise Exception("not implemented")
