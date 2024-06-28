import sys
import pandas

from dataloader.roa import RoADataLoader, RoAColumns


def view_financials(ticker: str, start: int, end: int) -> None:
    if end < start:
        sys.exit("Invalid parameters: start year is bigger than end year")

    dataloader = RoADataLoader()
    dataset_x, dataset_y = dataloader.get()

    dataset_x = dataset_x.loc[ticker]
    columns = [
        RoAColumns.TOTAL_ASSETS_COL,
        RoAColumns.NET_INCOME_COL,
        RoAColumns.get_col_mean(RoAColumns.TOTAL_ASSETS_COL),
    ]
    dataset_x = dataset_x.loc[
        (dataset_x.index >= start) & (dataset_x.index <= end), columns
    ]

    dataset_y = dataset_y.loc[ticker]
    dataset_y = dataset_y.loc[(dataset_y.index >= start) & (dataset_y.index <= end), :]
    result = pandas.concat([dataset_x, dataset_y], axis=1, join="outer")
    print("Numbers are represented in millions")
    print(result)
