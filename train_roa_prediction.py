from pandas import pandas
from collections import defaultdict

from metrics.metric import Metric
from models.model import Model
from trainer.trainer import Trainer
from logger.logger import Logger

from dataloader.roa import RoADataLoader, RoAColumns
from metrics.roa import RoAMetric
from models.dt_classifier import DTClassiferModel
from models.rf_classifier import RFClassifierModel
from models.hgb_classifier import HGBClassiferModel
from models.select_random import SelectRandomModel
from models.select_top import SelectTopModel


def train_roa_prediction() -> None:
    def get_metric(
        trainer: Trainer, model: Model, future_net_incomes: pandas.DataFrame
    ):
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

    # This predicts next 5 yr future return average from 2016
    dataset_x, dataset_y, future_net_incomes = predict_fixed_period_exceeds_threshold(
        year=2016, threshold=0.10
    )
    # This predicts next 5 yr future return average of all periods
    # dataset_x, dataset_y, future_net_incomes = predict_any_period_exceeds_threshold(
    #     0.10
    # )

    trainer = Trainer(dataset_x, dataset_y)
    logger = Logger("train_roa_prediction").get(__name__)

    metric_dict: defaultdict[str, list[Metric]] = defaultdict(list)
    for _ in range(20):
        model_dict = {
            "Select Random": SelectRandomModel(0.5),
            "Select Top": SelectTopModel(
                frac=0.5,
                cheatsheet=future_net_incomes,
                sort_by_col=RoAColumns.ROE_COl,
                ascending=False,
            ),
            "Decision Tree": DTClassiferModel(max_leaf_nodes=5),
            "Random Forest": RFClassifierModel(),
            "Histogram-gradient Boosting": HGBClassiferModel(),
        }

        trainer.train_test_split()
        for name, model in model_dict.items():
            metric = get_metric(trainer, model, future_net_incomes)
            logger.info(metric)
            # if name == "Random Forest":
            metric_dict[name].append(metric)

    for name, metrics in metric_dict.items():
        avg_metric = sum([m.value() for m in metrics]) / len(metrics)
        print(metrics[-1])
        print(
            f"\033[92m{name}: average return of portfolio is {avg_metric:.2f}%\n\033[0m"
        )
    # dt_model.visualize()
