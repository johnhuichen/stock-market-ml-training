import pandas
from typing import Self, Any


class Ticker:
    all_labels = [
        "id",
        "Name",
        "Exchange",
        "GicSector",
        "GicGroup",
    ]

    @classmethod
    def from_db(cls, document) -> Self:
        general = document["General"]
        data = {l: general[l] if l in general else "" for l in Ticker.all_labels}
        data["id"] = document["_id"]
        series = (
            pandas.Series(data=data, dtype=str)
            .fillna("")
            .map(lambda x: x.replace(",", "").strip())
        )

        return cls(series)

    @classmethod
    def to_csv_header(cls) -> str:
        return f"{','.join(Ticker.all_labels)}\n"

    def __init__(self, series: pandas.Series) -> None:
        self.series = series

    def __getitem__(self, key: str) -> Any:
        return self.series[key]

    def to_csv(self) -> str:
        return f"{','.join(self.series)}\n"


class TickerList:
    @classmethod
    def from_csv(cls, csv_file: str) -> Self:
        return cls(pandas.read_csv(csv_file, dtype="str"))

    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.dataframe = dataframe

    def __getitem__(self, key: str) -> Any:
        return self.dataframe[key]
