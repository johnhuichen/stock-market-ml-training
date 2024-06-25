from pathlib import Path
import pandas


class FutureNetIncomeDataLoader:
    TOTAL_ASSETS = "totalAssets3YrMean"
    NET_INCOME = "netIncome3YrMean"
    FUTURE_NET_INCOME = "futureNetIncome3YrMean"

    def __init__(self):
        self.financials_csv = Path(
            __file__ + "/../../data_source/financials.csv"
        ).resolve()
        dataframe = pandas.read_csv(self.financials_csv, index_col=[0, 1])

        # only include data with valid total assets 3 year average
        # represent numbers in millions for readability
        dataset_x = dataframe[
            ~dataframe[FutureNetIncomeDataLoader.TOTAL_ASSETS].isnull()
        ]
        dataset_x = dataset_x.xs(year_x, level=1) / 1e6

        # only include data with valid net income 3 year average
        # represent numbers in millions for readability
        all_returns = dataframe[
            ~dataframe[FutureNetIncomeDataLoader.NET_INCOME].isnull()
        ]
        all_returns = all_returns.xs(year_y + 2, level=1) / 1e6
        all_returns[FutureNetIncomeDataLoader.PREDICTION] = (
            all_returns[FutureNetIncomeDataLoader.NET_INCOME]
            / dataset_x[FutureNetIncomeDataLoader.TOTAL_ASSETS]
        )
        all_returns = all_returns[[FutureNetIncomeDataLoader.PREDICTION]]
        self.all_returns = all_returns

        dataset_y = (all_returns > threshold).astype(int)
        self.threshold = threshold
