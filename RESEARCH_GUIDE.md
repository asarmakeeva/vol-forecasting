# GARCH Volatility Forecasting Research Guide

## Overview

This repository contains comprehensive research comparing GARCH models with other volatility forecasting approaches. The research includes:

- **Classical Models:** GARCH(1,1), EGARCH(1,1), GJR-GARCH
- **Baseline Methods:** Historical volatility, EWMA
- **Evaluation:** Statistical tests, forecast accuracy, economic value
- **Visualization:** Publication-quality plots and diagnostics

## Quick Start

### 1. Run Complete Research Pipeline

```bash
python run_garch_research.py [TICKER] [START_DATE] [END_DATE]
```

Example:
```bash
python run_garch_research.py SPY 2015-01-01 2024-10-28
```

This will:
1. Download data from Yahoo Finance
2. Test stylized facts of returns
3. Compare GARCH model specifications
4. Fit models and generate diagnostics
5. Produce out-of-sample forecasts
6. Evaluate forecast accuracy
7. Run volatility targeting backtests
8. Generate publication-quality figures

### 2. Interactive Research Notebook

For exploratory analysis:
```bash
jupyter notebook research_garch_volatility.ipynb
```

The notebook contains:
- Detailed explanations of GARCH models
- Step-by-step implementation
- Mathematical formulations
- Comprehensive visualizations
- Extensions and future work

## Project Structure

```
vol-forecasting/
├── src/
│   ├── models/
│   │   ├── garch.py              # Enhanced GARCH implementations
│   │   ├── lstm.py               # Deep learning comparison
│   │   └── transformer.py
│   ├── data/
│   │   ├── features.py           # Volatility estimators
│   │   └── dataset.py
│   ├── eval/
│   │   ├── metrics.py            # Forecast evaluation
│   │   ├── backtests.py          # Vol-targeting strategy
│   │   └── plots.py              # Visualization utilities
│   └── research/
│       ├── garch_analysis.py     # Research analysis tools
│       └── __init__.py
├── research_garch_volatility.ipynb   # Interactive notebook
├── run_garch_research.py             # Quick start script
└── RESEARCH_GUIDE.md                 # This file
```

## Key Research Components

### 1. Model Implementations (`src/models/garch.py`)

```python
from models.garch import fit_garch, get_model_params, compare_garch_models

# Fit GARCH(1,1) with Student-t distribution
model = fit_garch(returns, kind='garch', distribution='studentst')

# Compare different specifications
comparison = compare_garch_models(returns)

# Generate rolling forecasts
forecasts = rolling_garch_forecast(returns, window=252, refit_freq=20)
```

**Supported Models:**
- `'garch'` - Standard GARCH(1,1)
- `'egarch'` - EGARCH (captures leverage effect)
- `'gjr'` - GJR-GARCH (asymmetric response)

**Distributions:**
- `'normal'` - Gaussian errors
- `'studentst'` - Student-t (heavy tails)
- `'skewstudent'` - Skewed Student-t

### 2. Research Analysis (`src/research/garch_analysis.py`)

```python
from research.garch_analysis import VolatilityComparison, GARCHDiagnostics

# Compare multiple forecasts
comp = VolatilityComparison(realized_vol)
comp.add_forecast('GARCH', garch_forecast)
comp.add_forecast('EWMA', ewma_forecast)

# Compute metrics
metrics = comp.compute_metrics()  # RMSE, MAE, QLIKE, R², MZ regression

# Diebold-Mariano test for forecast comparison
dm_test = comp.diebold_mariano_test('GARCH', 'EWMA', loss_func='qlike')

# Regime analysis
regime_results = comp.regime_analysis(regime_labels)

# Model diagnostics
diagnostics = GARCHDiagnostics.standardized_residuals_test(std_resid)
leverage_test = GARCHDiagnostics.leverage_effect_test(returns, volatility)
```

### 3. Visualizations (`src/eval/plots.py`)

```python
from eval.plots import (
    plot_volatility_comparison,
    plot_forecast_errors,
    plot_stylized_facts,
    plot_garch_diagnostics,
    plot_backtest_results
)

# Compare forecasts
fig = plot_volatility_comparison(
    actual=realized_vol,
    forecasts={'GARCH': garch_fcst, 'EWMA': ewma_fcst},
    highlight_periods={'COVID': ('2020-02-01', '2020-05-01')}
)

# Diagnostic plots
fig = plot_garch_diagnostics(garch_model.std_resid)

# Backtest results
fig = plot_backtest_results(backtest_dict)
```

### 4. Evaluation Metrics (`src/eval/metrics.py`, `src/eval/backtests.py`)

**Forecast Accuracy:**
- RMSE - Root Mean Squared Error
- MAE - Mean Absolute Error
- QLIKE - Quasi-Likelihood (robust to outliers)
- Mincer-Zarnowitz regression (unbiasedness test)
- Diebold-Mariano test (forecast comparison)

**Economic Value:**
- Volatility targeting strategy: `w_t = σ* / σ_t`
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Turnover costs (5 bps trade + 1 bps slippage)

## Research Workflow

### Step-by-Step Example

```python
import pandas as pd
import numpy as np
from models.garch import fit_garch, rolling_garch_forecast
from research.garch_analysis import VolatilityComparison, stylized_facts_summary
from eval.plots import plot_volatility_comparison

# 1. Load data
import yfinance as yf
df = yf.download('SPY', start='2015-01-01', end='2024-10-28')
returns = df['Adj Close'].pct_change().dropna()

# 2. Test stylized facts
facts = stylized_facts_summary(returns)
print(facts)

# 3. Fit GARCH model
train = returns[:'2020-12-31']
model = fit_garch(train * 100, kind='egarch', distribution='studentst')
print(model.summary())

# 4. Generate forecasts
forecasts = rolling_garch_forecast(returns, window=252, kind='egarch')

# 5. Evaluate
comp = VolatilityComparison(realized_vol)
comp.add_forecast('EGARCH', forecasts)
metrics = comp.compute_metrics()
print(metrics)

# 6. Visualize
fig = plot_volatility_comparison(realized_vol, {'EGARCH': forecasts})
```

## Methodology

### Walk-Forward Validation

We use expanding window walk-forward validation:
1. **Initial training:** Use first N years (e.g., 2015-2020)
2. **Test period:** Remaining data (e.g., 2021-2024)
3. **Refit frequency:** Monthly (every 20 trading days)
4. **Forecast horizon:** 1-day ahead

This prevents look-ahead bias and mimics real-world deployment.

### Volatility Estimation

We use the **Garman-Klass** estimator as a proxy for realized volatility:

```
GK_t = 0.5 * [ln(H_t/L_t)]² - (2*ln(2)-1) * [ln(C_t/O_t)]²
RV_t = sqrt(sum_{i=t-4}^{t} GK_i * 252)
```

This uses OHLC data to estimate intraday volatility more efficiently than close-to-close returns.

### Economic Evaluation

Volatility targeting strategy:
```
w_t = σ* / σ_{t|t-1}
```

Where:
- `w_t` = position size at time t (leverage)
- `σ*` = target volatility (default: 10% annualized)
- `σ_{t|t-1}` = volatility forecast

**Costs:**
- Transaction cost: 5 bps
- Slippage: 1 bps
- Max leverage: 2x

## Expected Results

### Stylized Facts (SPY 2015-2024)

- ✓ Volatility clustering (Ljung-Box p < 0.05 on squared returns)
- ✓ Heavy tails (Excess kurtosis ≈ 8-12)
- ✓ Leverage effect (Correlation(r_t, σ_t+1) < -0.1)
- ✓ Non-normal distribution (Jarque-Bera p < 0.001)

### GARCH(1,1) Typical Parameters

- ω (omega): ~0.01-0.05 (long-run variance)
- α (ARCH): ~0.05-0.15 (sensitivity to shocks)
- β (GARCH): ~0.80-0.90 (persistence)
- Persistence (α+β): ~0.95-0.99 (high persistence)
- Half-life: ~15-70 days

### Forecast Accuracy (Test Period)

Expected QLIKE scores (lower is better):
- GARCH(1,1): 0.45-0.50
- EGARCH(1,1): 0.43-0.48 (leverage effect helps)
- Historical Vol: 0.50-0.55
- EWMA: 0.45-0.50

### Economic Value (Sharpe Ratios)

Expected Sharpe ratios (net of costs):
- Vol-targeting (GARCH): 1.0-1.3
- Vol-targeting (EGARCH): 1.0-1.3
- Vol-targeting (EWMA): 0.9-1.2
- Buy & Hold: 0.7-0.9

## Advanced Topics

### 1. Distribution Selection

Test different error distributions:
```python
from models.garch import compare_garch_models

comparison = compare_garch_models(returns, models=[
    ('garch', 'normal'),
    ('garch', 'studentst'),
    ('egarch', 'studentst'),
])
print(comparison.sort_values('AIC'))
```

**Rule of thumb:** Student-t usually preferred for financial returns (heavy tails).

### 2. Regime Analysis

Classify volatility regimes and evaluate by regime:
```python
def classify_regime(rv):
    if rv < 0.15:
        return 'Low'
    elif rv < 0.25:
        return 'Medium'
    else:
        return 'High'

regimes = realized_vol.apply(classify_regime)
regime_results = comp.regime_analysis(regimes)
```

### 3. Model Diagnostics

Check if GARCH adequately captures volatility dynamics:
```python
from research.garch_analysis import GARCHDiagnostics

diagnostics = GARCHDiagnostics.standardized_residuals_test(model.std_resid)

# Should see:
# - No autocorrelation in residuals (p > 0.05)
# - No remaining ARCH effects (p > 0.05 for squared residuals)
# - Possibly non-normal (Student-t better than Gaussian)
```

### 4. Statistical Tests

**Diebold-Mariano Test:** Compare two forecasts
```python
dm_result = comp.diebold_mariano_test('GARCH', 'EWMA', loss_func='qlike')
if dm_result['significant']:
    print(f"Better model: {dm_result['better_model']}")
```

**Mincer-Zarnowitz Regression:** Test forecast efficiency
```
RV_t = a + b * Forecast_t + error_t
```
- Ideal: a=0 (unbiased), b=1 (efficient), R²=high

## Troubleshooting

### Common Issues

**1. Convergence Failures**
```python
# Scale returns by 100 for numerical stability
model = fit_garch(returns * 100, kind='garch')
```

**2. Insufficient Data**
```python
# Need at least 1 year (252 days) for reliable estimates
if len(returns) < 252:
    raise ValueError("Need at least 252 observations")
```

**3. Non-stationary Model**
```python
params = get_model_params(model)
if params['persistence'] >= 1.0:
    print("Warning: Non-stationary model (α+β ≥ 1)")
```

**4. Remaining ARCH Effects**
```python
# If diagnostics show remaining ARCH, try:
# - Higher order GARCH(p,q)
# - Different distribution (Student-t)
# - Check for structural breaks
```

## References

### Key Papers

1. **Bollerslev (1986)**: Generalized autoregressive conditional heteroskedasticity
2. **Nelson (1991)**: EGARCH model
3. **Hansen & Lunde (2005)**: Forecast comparison of volatility models
4. **Diebold & Mariano (1995)**: Comparing predictive accuracy

### Software Documentation

- [arch](https://arch.readthedocs.io/): GARCH models in Python
- [statsmodels](https://www.statsmodels.org/): Statistical tests
- [yfinance](https://github.com/ranaroussi/yfinance): Yahoo Finance data

## Contributing

To extend this research:

1. Add new models in `src/models/`
2. Add evaluation metrics in `src/eval/metrics.py`
3. Add visualizations in `src/eval/plots.py`
4. Update notebook with new findings

## Citation

If you use this research in academic work:

```bibtex
@misc{garch_volatility_research,
  title={GARCH Models for Volatility Forecasting: A Comprehensive Study},
  author={Research Team},
  year={2024},
  url={https://github.com/username/vol-forecasting}
}
```

## License

MIT License - See LICENSE file for details.

---

**Last Updated:** October 2024
**Contact:** [Your contact info]
