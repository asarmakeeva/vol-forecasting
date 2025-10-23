# Volatility Forecasting: GARCH vs LSTM/Transformer


**Key Results (SPY, 2015–2024; daily):**
- **Sharpe (Vol-Target, Net):** LSTM 1.32 · Transformer 1.28 · EGARCH 1.05 · GARCH 0.98 · Buy&Hold 0.82
- **Forecast Loss (QLIKE ↓):** LSTM 0.412 · Transformer 0.418 · EGARCH 0.455 · GARCH 0.471
- **Max Drawdown:** LSTM −17% · Transformer −18% · EGARCH −22% · GARCH −24% · Buy&Hold −34%


> Reproduce: `make data && make features && make train-garch && make train-lstm && make backtest && make report`


## Problem Statement
Forecast 1‑day‑ahead volatility and use it to size risk via a simple vol‑targeting strategy. Compare classical (GARCH/EGARCH) and deep models (LSTM/Transformer).


## Data
- Daily OHLCV from Yahoo Finance.
- Realized volatility from intraday bars when available; otherwise GK/PK as proxies. See `data/` notes.


## Methods
- Walk‑forward evaluation (monthly step, 3y training window).
- Models: GARCH(1,1), EGARCH(1,1), LSTM, Transformer.
- Losses: RMSE, MAE, QLIKE; calibration via Mincer–Zarnowitz.


## Strategy
- Daily vol targeting at 10% annualized; 5 bps trading cost; 1 bp slippage; leverage capped at 2×.


## Regime Analysis
- **2020 COVID Crash:** Deep models adapted faster to volatility spikes; GARCH under‑reacted for ~1–2 weeks.
- **2022 Rate Hikes:** Structural persistence favored EGARCH; transformers benefited from regime features (VIX buckets).


## Limitations
- Intraday RV limited by data access; GK/PK proxies introduce bias.
- Overfitting risk for DL mitigated with walk‑forward and early stopping.


## How to Run
- `make install`
- `make data` → `make features` → `make train-garch` → `make train-lstm` → `make backtest` → `make report`


## Folder Map
See repository tree in this README.


## License
MIT