#!/usr/bin/env python3
"""
Quick Start: GARCH Volatility Research
======================================

Run comprehensive GARCH model analysis and comparison.

Usage:
    python run_garch_research.py [ticker] [start_date] [end_date]

Example:
    python run_garch_research.py SPY 2015-01-01 2024-10-28
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

from models.garch import fit_garch, get_model_params, compare_garch_models, rolling_garch_forecast
from data.features import realized_vol_from_daily
from eval.metrics import qlike
from eval.backtests import vol_target_weights, run_backtest
from eval.plots import (plot_volatility_comparison, plot_forecast_errors,
                       plot_stylized_facts, plot_garch_diagnostics,
                       plot_backtest_results)
from research.garch_analysis import (VolatilityComparison, GARCHDiagnostics,
                                    stylized_facts_summary, format_research_table)


def load_data(ticker='SPY', start='2015-01-01', end='2024-10-28'):
    """Load data from Yahoo Finance"""
    import yfinance as yf
    print(f"Downloading {ticker} data from {start} to {end}...")

    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df['returns'] = df['adj close'].pct_change()
    df['log_returns'] = np.log(df['adj close'] / df['adj close'].shift(1))
    df['rv'] = realized_vol_from_daily(df)

    print(f"✓ Downloaded {len(df)} days of data\n")
    return df.dropna()


def main():
    """Main research pipeline"""
    # Parse arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2015-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2024-10-28'

    print("=" * 80)
    print("GARCH VOLATILITY FORECASTING RESEARCH")
    print("=" * 80)
    print()

    # Load data
    df = load_data(ticker, start_date, end_date)

    # Split into train/test
    train_end = '2020-12-31'
    test_start = '2021-01-01'

    returns_train = df.loc[:train_end, 'returns'].dropna()
    returns_test = df.loc[test_start:, 'returns'].dropna()

    print(f"Training: {returns_train.index[0]} to {returns_train.index[-1]} ({len(returns_train)} days)")
    print(f"Testing:  {returns_test.index[0]} to {returns_test.index[-1]} ({len(returns_test)} days)")
    print()

    # =================================================================
    # 1. STYLIZED FACTS
    # =================================================================
    print("=" * 80)
    print("1. TESTING STYLIZED FACTS OF RETURNS")
    print("=" * 80)

    facts = stylized_facts_summary(df['returns'])

    print(f"\nReturn Statistics:")
    print(f"  Mean (annualized):     {facts['mean_return']*252*100:>8.2f}%")
    print(f"  Volatility (annual):   {facts['volatility']*np.sqrt(252)*100:>8.2f}%")
    print(f"  Skewness:              {facts['skewness']:>8.3f}")
    print(f"  Excess Kurtosis:       {facts['kurtosis']:>8.3f}")

    print(f"\nHypothesis Tests:")
    print(f"  ✓ Volatility clustering:  {'YES' if facts['volatility_clustering'] else 'NO'}")
    print(f"  ✓ Heavy tails:            {'YES' if facts['heavy_tails'] else 'NO'}")
    print(f"  ✓ Normal distribution:    {'YES' if facts['normal_distribution'] else 'NO'}")
    print()

    # Plot stylized facts
    fig1 = plot_stylized_facts(df['returns'], figsize=(15, 10))
    plt.savefig('stylized_facts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: stylized_facts.png\n")
    plt.close()

    # =================================================================
    # 2. MODEL COMPARISON
    # =================================================================
    print("=" * 80)
    print("2. COMPARING GARCH MODEL SPECIFICATIONS")
    print("=" * 80)

    model_comparison = compare_garch_models(returns_train)
    print("\n" + model_comparison.to_string(index=False))
    print("\n(Lower AIC/BIC is better)\n")

    # =================================================================
    # 3. FIT BEST MODELS
    # =================================================================
    print("=" * 80)
    print("3. FITTING MODELS ON TRAINING DATA")
    print("=" * 80)

    print("\nFitting GARCH(1,1)...")
    garch_model = fit_garch(returns_train * 100, kind='garch', distribution='studentst')
    garch_params = get_model_params(garch_model)

    print(f"\n  GARCH(1,1) Parameters:")
    print(f"    ω (omega):          {garch_params['omega']:.6f}")
    print(f"    α (ARCH effect):    {garch_params['alpha[1]']:.4f}")
    print(f"    β (GARCH effect):   {garch_params['beta[1]']:.4f}")
    print(f"    Persistence (α+β):  {garch_params['persistence']:.4f}")
    print(f"    Half-life:          {garch_params['half_life']:.1f} days")

    print("\nFitting EGARCH(1,1)...")
    egarch_model = fit_garch(returns_train * 100, kind='egarch', distribution='studentst')
    egarch_params = get_model_params(egarch_model)

    print(f"\n  EGARCH(1,1) Parameters:")
    print(f"    γ (leverage):       {egarch_params.get('gamma[1]', 0):.4f}")
    if egarch_params.get('gamma[1]', 0) < 0:
        print(f"    ✓ Leverage effect detected: negative shocks increase volatility more")

    # Model diagnostics
    print("\n  GARCH Diagnostics:")
    garch_diag = GARCHDiagnostics.standardized_residuals_test(garch_model.std_resid)
    print(f"    No autocorrelation:     {'✓' if garch_diag['no_autocorr'] else '✗'}")
    print(f"    No remaining ARCH:      {'✓' if garch_diag['no_arch'] else '✗'}")
    print(f"    Normal residuals:       {'✓' if garch_diag['normal'] else '✗'}")

    # Plot diagnostics
    fig2 = plot_garch_diagnostics(garch_model.std_resid)
    plt.savefig('garch_diagnostics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: garch_diagnostics.png\n")
    plt.close()

    # =================================================================
    # 4. OUT-OF-SAMPLE FORECASTING
    # =================================================================
    print("=" * 80)
    print("4. GENERATING OUT-OF-SAMPLE FORECASTS")
    print("=" * 80)

    print("\nGenerating rolling forecasts (this may take a few minutes)...")

    # Generate forecasts for test period
    test_rv = df.loc[test_start:, 'rv']

    # GARCH forecasts
    garch_fcst = rolling_garch_forecast(df.loc[:, 'returns'], window=1260, kind='garch', refit_freq=20)
    egarch_fcst = rolling_garch_forecast(df.loc[:, 'returns'], window=1260, kind='egarch', refit_freq=20)

    # Baseline forecasts
    hist_vol = df['returns'].rolling(20).std() * np.sqrt(252)
    ewma_vol = df['returns'].ewm(halflife=10).std() * np.sqrt(252)

    print("✓ Forecasts generated\n")

    # =================================================================
    # 5. FORECAST EVALUATION
    # =================================================================
    print("=" * 80)
    print("5. EVALUATING FORECAST ACCURACY")
    print("=" * 80)

    comp = VolatilityComparison(test_rv)
    comp.add_forecast('GARCH(1,1)', garch_fcst)
    comp.add_forecast('EGARCH(1,1)', egarch_fcst)
    comp.add_forecast('Historical Vol', hist_vol.loc[test_start:])
    comp.add_forecast('EWMA', ewma_vol.loc[test_start:])

    metrics = comp.compute_metrics()
    print("\n" + format_research_table(metrics[['Model', 'RMSE', 'MAE', 'QLIKE', 'R²', 'Bias']]))

    print("\nInterpretation:")
    print("  - Lower RMSE/MAE/QLIKE is better (smaller forecast errors)")
    print("  - Higher R² indicates better fit")
    print("  - Bias should be close to zero (unbiased forecast)")
    print()

    # Diebold-Mariano test
    if 'GARCH(1,1)' in comp.forecasts and 'EWMA' in comp.forecasts:
        dm_result = comp.diebold_mariano_test('GARCH(1,1)', 'EWMA', loss_func='qlike')
        print(f"Diebold-Mariano Test: GARCH vs EWMA")
        print(f"  Test statistic: {dm_result['DM_statistic']:.3f}")
        print(f"  P-value:        {dm_result['p_value']:.4f}")
        print(f"  Result:         {'Significantly different' if dm_result['significant'] else 'Not significantly different'}")
        print(f"  Better model:   {dm_result['better_model']}")
        print()

    # Plot comparisons
    fig3 = plot_volatility_comparison(
        test_rv,
        {'GARCH': garch_fcst, 'EGARCH': egarch_fcst, 'EWMA': ewma_vol.loc[test_start:]},
        title=f"{ticker} Volatility Forecasts (Test Period)"
    )
    plt.savefig('forecast_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: forecast_comparison.png\n")
    plt.close()

    # =================================================================
    # 6. ECONOMIC VALUE: VOLATILITY TARGETING
    # =================================================================
    print("=" * 80)
    print("6. EVALUATING ECONOMIC VALUE (VOLATILITY TARGETING)")
    print("=" * 80)

    print("\nRunning backtests...")

    backtests = {}

    # GARCH strategy
    garch_weights = vol_target_weights(garch_fcst, sigma_star=0.10, w_max=2.0)
    backtests['GARCH'] = run_backtest(returns_test, garch_weights, tc_bps=5, slip_bps=1)

    # EWMA strategy
    ewma_weights = vol_target_weights(ewma_vol.loc[test_start:], sigma_star=0.10, w_max=2.0)
    backtests['EWMA'] = run_backtest(returns_test, ewma_weights, tc_bps=5, slip_bps=1)

    # Buy & Hold
    bh_weights = pd.Series(1.0, index=returns_test.index)
    backtests['Buy & Hold'] = run_backtest(returns_test, bh_weights, tc_bps=5, slip_bps=1)

    # Results table
    bt_summary = pd.DataFrame({
        'Strategy': [name for name in backtests.keys()],
        'Sharpe Ratio': [bt['sharpe'] for bt in backtests.values()],
        'Max Drawdown (%)': [bt['max_drawdown'] * 100 for bt in backtests.values()],
        'Avg Turnover': [bt['turnover'] for bt in backtests.values()],
        'Final Equity': [bt['equity'].iloc[-1] for bt in backtests.values()]
    })

    print("\n" + format_research_table(bt_summary))

    # Plot results
    fig4 = plot_backtest_results(backtests)
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: backtest_results.png\n")
    plt.close()

    # =================================================================
    # SUMMARY
    # =================================================================
    print("=" * 80)
    print("RESEARCH SUMMARY")
    print("=" * 80)

    print(f"\n✓ Data: {ticker} ({start_date} to {end_date})")
    print(f"✓ Training: {len(returns_train)} days")
    print(f"✓ Testing: {len(returns_test)} days")

    print("\n✓ Stylized Facts:")
    print(f"  - Volatility clustering: {'Confirmed' if facts['volatility_clustering'] else 'Not found'}")
    print(f"  - Heavy tails: {'Confirmed' if facts['heavy_tails'] else 'Not found'}")

    print("\n✓ Best Model (by AIC):")
    best_model = model_comparison.iloc[0]
    print(f"  - {best_model['Model']} with {best_model['Distribution']} distribution")
    print(f"  - AIC: {best_model['AIC']:.2f}")

    print("\n✓ Forecast Accuracy (Test Period):")
    best_forecast = metrics.nsmallest(1, 'QLIKE').iloc[0]
    print(f"  - Best model: {best_forecast['Model']}")
    print(f"  - QLIKE: {best_forecast['QLIKE']:.4f}")
    print(f"  - R²: {best_forecast['R²']:.4f}")

    print("\n✓ Economic Value (Sharpe Ratio):")
    best_strategy = bt_summary.nsmallest(1, 'Sharpe Ratio', keep='last').iloc[-1]
    print(f"  - Best strategy: {best_strategy['Strategy']}")
    print(f"  - Sharpe: {best_strategy['Sharpe Ratio']:.2f}")
    print(f"  - Max DD: {best_strategy['Max Drawdown (%)']:.2f}%")

    print("\n✓ Generated Figures:")
    print("  - stylized_facts.png")
    print("  - garch_diagnostics.png")
    print("  - forecast_comparison.png")
    print("  - backtest_results.png")

    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
