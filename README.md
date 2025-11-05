# Volatility Forecasting: GARCH vs LSTM

Comprehensive comparison of classical GARCH models and deep learning LSTM for financial volatility forecasting.

## 📊 Key Results (SPY, 2015-2024)

| Model | QLIKE ↓ | Sharpe ↑ | Max DD |
|-------|---------|----------|--------|
| **LSTM** | **0.412** | **1.32** | -17% |
| EGARCH | 0.455 | 1.05 | -22% |
| GARCH | 0.471 | 0.98 | -24% |
| Buy&Hold | - | 0.82 | -34% |

**LSTM achieves ~10% better forecast accuracy and +0.30 Sharpe improvement**

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run comparison (15-30 min)
make compare

# Or directly
python compare_garch_lstm.py SPY 2015-01-01 2024-10-28
```

---

## 📁 Project Structure

```
vol-forecasting/
├── compare_garch_lstm.py         # Main comparison script ⭐
├── Makefile                       # Convenient commands
├── data/
│   ├── raw/                       # Downloaded OHLCV
│   └── processed/                 # Features + targets
├── notebooks/
│   ├── 02_garch_baselines.ipynb  # ✓ Complete GARCH research
│   └── README.md                  # Notebook descriptions
├── src/
│   ├── config.py                  # Configuration
│   ├── data/                      # Data processing
│   ├── models/                    # GARCH, LSTM, Transformer
│   ├── eval/                      # Metrics, plots, backtest
│   ├── utils/                     # Timesplits, I/O
│   └── research/                  # Analysis tools
└── tests/                         # Unit tests
```

---

## 💻 Usage

```bash
make help          # Show all commands
make compare       # Run GARCH vs LSTM comparison
make notebooks     # Launch Jupyter notebooks
make test          # Run tests
```

**Try other assets:**
```bash
python compare_garch_lstm.py QQQ 2015-01-01 2024-10-28
python compare_garch_lstm.py GLD 2015-01-01 2024-10-28
```

---

## 🔬 Methodology

- **Walk-forward validation** (expanding window, monthly refit)
- **Important lags**: [1, 2, 6, 11, 16] (from feature analysis)
- **Evaluation**: RMSE, MAE, QLIKE, Diebold-Mariano test
- **Backtest**: Vol-targeting (10% target, 6bps costs, 2x max leverage)

---

## ⚙️ Configuration

Edit `src/config.py`:
```python
IMPORTANT_LAGS = [1, 2, 6, 11, 16]
GARCH_MODELS = ['garch', 'egarch']
LSTM_SEQ_LEN = 30
VOL_TARGET = 0.10
```

---

## 📚 Key Findings

**Use GARCH when:**
- Limited data (< 2 years)
- Need interpretability
- Stable markets

**Use LSTM when:**
- Large dataset (3+ years)
- Regime changes frequent
- Multiple data sources

---

## 📄 License

MIT License
