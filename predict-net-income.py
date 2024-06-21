from trainers.net_income_trainer import NetIncomeTrainer
from models.decision_tree import DecisionTree
from models.random_select import RandomSelect
from models.top_percent import TopPercent

trainer = NetIncomeTrainer(2016)

model = DecisionTree()
metric = trainer.train(model)
print(metric)
model.visualize()

model = RandomSelect(1.0)
metric = trainer.train(model)
print(metric)

model = RandomSelect(0.5)
metric = trainer.train(model)
print(metric)

model = TopPercent(0.5, NetIncomeTrainer.PREDICTION)
metric = trainer.train(model)
print(metric)
