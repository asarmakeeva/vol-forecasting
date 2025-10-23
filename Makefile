install:
pip install -r requirements.txt


data:
python -m src.data.download --tickers SPY QQQ AAPL MSFT --start 2015-01-01
features:
python -m src.data.features --config configs/base.yaml
train-garch:
python -m src.models.garch --config configs/garch.yaml
train-lstm:
python -m src.models.lstm --config configs/lstm.yaml
train-transformer:
python -m src.models.transformer --config configs/transformer.yaml
backtest:
python -m src.eval.backtest --config configs/backtest.yaml
report:
jupyter nbconvert --to html --execute notebooks/04_backtest_report.ipynb