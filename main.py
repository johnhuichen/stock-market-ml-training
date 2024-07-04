#!/usr/bin/env python3

import sys
import argparse
from datetime import datetime

from prepare_data import prepare_data
from train_roa_prediction import train_roa_prediction
from view_financials import view_financials


class Commands:
    PREPARE_DATA = "prepare_data"
    TRAIN_ROE = "train_roe"
    VIEW_FINANCIALS = "view_financials"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="stock_market_ml_training",
        description="Machine Learning Training in Stock Market",
    )
    subparsers = parser.add_subparsers(dest="cmd", help="Base Comands", required=True)

    parser_prepare_data = subparsers.add_parser(
        Commands.PREPARE_DATA, help="Prepare data from datasource"
    )
    parser_train_roe = subparsers.add_parser(
        Commands.TRAIN_ROE, help=f"Train and Predict Return on Assets"
    )

    parser_view_financials = subparsers.add_parser(
        Commands.VIEW_FINANCIALS, help=f"View Ticker Financials Data"
    )
    parser_view_financials.add_argument("ticker", type=str, help="Ticker Symbol")
    parser_view_financials.add_argument("start", type=int, help="Start Year")
    year_now = datetime.now().year
    parser_view_financials.add_argument(
        "-e", "--end", type=int, help="End Year", default=year_now
    )
    # parser_view_financials.add_argument(
    #     "-c",
    #     "--columns",
    #     type=str,
    #     nargs="+",
    #     help="Data Columns",
    #     default=[
    #         RoAColumns.TOTAL_ASSETS_COL,
    #         RoAColumns.NET_INCOME_COL,
    #         RoAColumns.get_col_mean(RoAColumns.TOTAL_ASSETS_COL),
    #     ],
    # )

    args = parser.parse_args()

    match args.cmd:
        case Commands.PREPARE_DATA:
            prepare_data()
        case Commands.TRAIN_ROE:
            train_roa_prediction()
        case Commands.VIEW_FINANCIALS:
            view_financials(
                ticker=args.ticker,
                start=args.start,
                end=args.end,
            )
        case _:
            sys.exit("Unexpected error")
