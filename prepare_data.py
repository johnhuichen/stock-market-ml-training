from pathlib import Path

from datasource.datasource import DataCsv


def prepare_data():
    answer = input("Continue? Enter 'yes' to proceed\n").lower()

    if answer == "yes":
        # remove csv files in dataloader
        for item in Path("dataloader").iterdir():
            if item.suffix == ".csv":
                item.unlink()

        data_csv = DataCsv()

        data_csv.save_tickers()
        data_csv.save_financials()
