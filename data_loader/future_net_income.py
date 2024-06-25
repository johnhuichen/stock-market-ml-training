from pathlib import Path
from typing import Tuple
import pandas


class FutureNetIncomeDataLoader:
    TOTAL_ASSETS_MEAN = "totalAssetsMean"
    NET_INCOME_MEAN = "netIncomeMean"
    NET_INCOME_MEAN_FUTURE = "netIncomeMeanFuture"
    RETURN_FUTURE = "returnFuture"

    def __init__(self, rolling_window: int = 3, year_to_target: int = 7):
        self.rolling_window = rolling_window
        self.year_to_target = year_to_target

        TOTAL_ASSETS_MEAN = FutureNetIncomeDataLoader.TOTAL_ASSETS_MEAN
        NET_INCOME_MEAN = FutureNetIncomeDataLoader.NET_INCOME_MEAN
        NET_INCOME_MEAN_FUTURE = FutureNetIncomeDataLoader.NET_INCOME_MEAN_FUTURE
        RETURN_FUTURE = FutureNetIncomeDataLoader.RETURN_FUTURE

        self.financials_csv = Path(
            __file__ + "/../../data_source/financials.csv"
        ).resolve()

        financials = pandas.read_csv(self.financials_csv, index_col=[0, 1])
        # represent numbers in millions
        financials = financials / 1e6
        financials_rolling_mean = pandas.DataFrame(
            financials.rolling(rolling_window).mean()
        ).rename(columns=lambda col: f"{col}Mean")
        financials = financials.merge(
            financials_rolling_mean, left_index=True, right_index=True
        )

        dataset_x = financials[~financials[TOTAL_ASSETS_MEAN].isnull()]

        dataset_y = financials.rename(index=lambda y: y - year_to_target, level=1)
        dataset_y = dataset_y[~dataset_y[NET_INCOME_MEAN].isnull()]
        dataset_y = dataset_y.loc[:, [NET_INCOME_MEAN]].rename(
            columns={NET_INCOME_MEAN: NET_INCOME_MEAN_FUTURE}
        )
        dataset_y[RETURN_FUTURE] = (
            dataset_y[NET_INCOME_MEAN_FUTURE] / dataset_x[TOTAL_ASSETS_MEAN]
        )

        self.dataset_x = dataset_x.loc[dataset_x.index.isin(dataset_y.index)]
        self.dataset_y = dataset_y.loc[dataset_y.index.isin(dataset_x.index)]

    def get(self) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        return self.dataset_x, self.dataset_y
