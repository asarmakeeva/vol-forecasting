# Notebooks

Sequential notebooks for volatility forecasting research:

## 01_exploratory_data.ipynb
- Data download and exploration
- Descriptive statistics
- Stylized facts of returns
- Volatility clustering visualization
- Leverage effect analysis

**Status:** To be developed (use `compare_garch_lstm.py` for now)

## 02_garch_baselines.ipynb ✓
- Comprehensive GARCH research
- GARCH(1,1) and EGARCH(1,1) models
- Model diagnostics and parameter interpretation
- Walk-forward forecasting
- Comparison with baselines

**Status:** Complete - contains full GARCH research

## 03_lstm_transformer.ipynb
- Feature engineering for deep learning
- LSTM architecture and training
- Transformer model (future work)
- Hyperparameter tuning
- Model interpretation

**Status:** To be developed (use `compare_garch_lstm.py` for now)

## 04_backtest_report.ipynb
- Volatility targeting strategy
- Comparative backtest results
- Statistical significance tests
- Economic value analysis
- Final conclusions

**Status:** To be developed (use `compare_garch_lstm.py` for complete analysis)

---

## Quick Start

Instead of notebooks, you can use the command-line tools:

```bash
# Full GARCH vs LSTM comparison
python compare_garch_lstm.py SPY 2015-01-01 2024-10-28

# GARCH research only
python run_garch_research.py SPY 2015-01-01 2024-10-28

# Or use Make
make compare
```

The notebooks provide an interactive exploration environment, while the scripts offer automated end-to-end pipelines.
