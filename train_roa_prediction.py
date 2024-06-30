from pandas import pandas

from dataloader.roa import RoADataLoader, RoAColumns
from metrics.roa import RoAMetric
from models.dt_classifier import DTClassiferModel
from models.rf_classifier import RFClassifierModel
from models.hgb_classifier import HGBClassiferModel
from models.select_random import SelectRandomModel
from models.select_top import SelectTopModel
from trainer.trainer import Trainer


def train_roa_prediction() -> None:
    def get_metric(trainer, model, future_net_incomes):
        predictions, val_y = trainer.train(model)
        return RoAMetric(
            model=model,
            predictions=predictions,
            val_y=val_y,
            future_net_incomes=future_net_incomes,
        )

    def predict_fixed_period_exceeds_threshold(
        year: int, threshold: float
    ) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        dataloader = RoADataLoader()
        dataset_x, future_net_incomes = dataloader.get()

        dataset_x = dataset_x.xs(year, level=1)
        future_net_incomes = future_net_incomes.xs(year, level=1)

        dataset_y = future_net_incomes.loc[:, [RoAColumns.ROE_COl]] > threshold
        return dataset_x, dataset_y, future_net_incomes

    def predict_any_period_exceeds_threshold(
        threshold: float,
    ) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        dataloader = RoADataLoader()
        dataset_x, future_net_incomes = dataloader.get()

        dataset_y = future_net_incomes.loc[:, [RoAColumns.ROE_COl]] > threshold
        return dataset_x, dataset_y, future_net_incomes

    # This predicts next 10 yr future return average from 2001
    # dataset_x, dataset_y, future_net_incomes = predict_fixed_period_exceeds_threshold(
    #     year=2016, threshold=0.10
    # )
    # This predicts next 10 yr future return average of all periods
    dataset_x, dataset_y, future_net_incomes = predict_any_period_exceeds_threshold(
        0.10
    )

    trainer = Trainer(dataset_x, dataset_y)

    dt_model = DTClassiferModel(max_leaf_nodes=5)
    rf_model = RFClassifierModel()
    hgb_model = HGBClassiferModel()
    select_top_model = SelectTopModel(
        frac=0.5,
        cheatsheet=future_net_incomes,
        sort_by_col=RoAColumns.ROE_COl,
        ascending=False,
    )
    select_random_model = SelectRandomModel(0.5)

    print(get_metric(trainer, select_random_model, future_net_incomes))
    print(get_metric(trainer, select_top_model, future_net_incomes))
    print(get_metric(trainer, dt_model, future_net_incomes))
    print(get_metric(trainer, rf_model, future_net_incomes))
    print(get_metric(trainer, hgb_model, future_net_incomes))
    # dt_model.visualize()
