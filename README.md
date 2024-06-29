# Machine Learning Application in Stock Market

# Table of Contents

1. [Get Started](#get-started)
1. [Road Map](#roadmap)

# Get Started

## Install Python 3.12

The easiest way to create an environment with Python 3.12 is to use conda

(Click on [Conda Doc](https://conda.io/) for how to install)

Once conda is installed, run

```shell
conda create --name stock-market-ml-training python=3.12 && conda activate stock-market-ml-training
```

## Install Python Dependencies

```shell
pip install -r requirements.txt
```

## Extract data from datasource/data.zip

```shell
unzip datasource/data.zip
```

## Train prediction of return-on-assets(RoA)

```python
python main.py train_row
```

## View financials of a ticker

```python
python main.py view_financials AAPL.US 2000
```

# Road Map

1. Add unit tests to ensure dataloader, models, metrics and train are working correctly
1. Add unit tests to mock db connection and ensure datasource is working correctly
1. Analyze tickers that give return on assets that are clearly outliers. An example is BPT.US which gives more than 150 times return on assets in some years
