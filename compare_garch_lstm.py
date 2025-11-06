#!/usr/bin/env python3
"""
GARCH vs LSTM: Comprehensive Volatility Forecasting Comparison
==============================================================

Compare classical GARCH models with deep learning LSTM for volatility forecasting.

Usage:
    python compare_garch_lstm.py [TICKER] [START_DATE] [END_DATE]

Example:
    python compare_garch_lstm.py SPY 2015-01-01 2024-10-28
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

# GARCH imports
from models.garch import fit_garch, rolling_garch_forecast, get_model_params

# LSTM imports
from models.lstm import rolling_lstm_forecast

# Feature engineering
from data.features import (
    realized_vol_from_daily,
    create_volatility_features,
    select_lstm_features,
    normalize_features
)

# Evaluation
from eval.metrics import qlike
from eval.backtest import vol_target_weights, run_backtest
from eval.plots import (
    plot_volatility_comparison,
    plot_forecast_errors,
    plot_scatter_comparison,
    plot_backtest_results
)

# Research analysis
from research.garch_analysis import VolatilityComparison, stylized_facts_summary

def load_data(ticker='SPY', start='2015-01-01', end='2024-10-28'):
    import yfinance as yf
    print(f"\nDownloading {ticker} data from {start} to {end}...")

    # Force columns grouped by field (not by ticker) to avoid MultiIndex;
    # still handle MultiIndex defensively just in case.
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        group_by='column',   # ensures Open/High/Low/Close/Adj Close/Volume
        auto_adjust=False
    )

    # Flatten any MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize names to lowercase strings
    df.columns = [str(c).lower() for c in df.columns]

    # Prefer 'close'; fall back to 'adj close' if needed
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']

    # Basic features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Realized volatility
    df['rv'] = realized_vol_from_daily(df)

    print(f" Downloaded {len(df)} days of data\n")
    return df.dropna()



def prepare_lstm_data(df_raw):
    """
    Prepare comprehensive features for LSTM

    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw OHLCV data

    Returns:
    --------
    features_df : pd.DataFrame
        Feature DataFrame with 'rv' target
    """
    print("Creating features for LSTM...")

    # Create all features
    features = create_volatility_features(df_raw)

    print(f"✓ Created {len(features.columns)} features")
    print(f"✓ Valid samples: {len(features)}")

    return features


def main():
    """Main comparison pipeline"""
    # Parse arguments
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2015-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2024-10-28'

    print("=" * 80)
    print("GARCH vs LSTM: VOLATILITY FORECASTING COMPARISON")
    print("=" * 80)

    # =================================================================
    # 1. LOAD DATA
    # =================================================================
    df = load_data(ticker, start_date, end_date)

    # Split into train/test
    train_end = '2020-12-31'
    test_start = '2021-01-01'

    print(f"Training period: {df.loc[:train_end].index[0].date()} to {df.loc[:train_end].index[-1].date()}")
    print(f"Testing period:  {df.loc[test_start:].index[0].date()} to {df.loc[test_start:].index[-1].date()}")
    print()

    # =================================================================
    # 2. STYLIZED FACTS
    # =================================================================
    print("=" * 80)
    print("1. ANALYZING STYLIZED FACTS")
    print("=" * 80)

    facts = stylized_facts_summary(df['returns'])

    print(f"\n✓ Volatility clustering: {'YES' if facts['volatility_clustering'] else 'NO'}")
    print(f"✓ Heavy tails:           {'YES' if facts['heavy_tails'] else 'NO'}")
    print(f"✓ Mean return:           {facts['mean_return']*252*100:.2f}% annualized")
    print(f"✓ Volatility:            {facts['volatility']*np.sqrt(252)*100:.2f}% annualized")
    print()

    # =================================================================
    # 3. GARCH MODELS
    # =================================================================
    print("=" * 80)
    print("2. FITTING GARCH MODELS")
    print("=" * 80)

    print("\nGenerating GARCH forecasts (this may take a few minutes)...")

    # Generate GARCH forecasts
    garch_fcst = rolling_garch_forecast(
        df['returns'],
        window=1260,  # 5 years
        kind='garch',
        refit_freq=20  # Monthly
    )

    egarch_fcst = rolling_garch_forecast(
        df['returns'],
        window=1260,
        kind='egarch',
        refit_freq=20
    )

    print(f"✓ GARCH forecasts:  {len(garch_fcst)} predictions")
    print(f"✓ EGARCH forecasts: {len(egarch_fcst)} predictions")
    print()

    # =================================================================
    # 4. LSTM MODEL
    # =================================================================
    print("=" * 80)
    print("3. TRAINING LSTM MODEL")
    print("=" * 80)

    # Prepare features
    features_df = prepare_lstm_data(df)

    # Select features for LSTM
    feature_cols = select_lstm_features(features_df, correlation_threshold=0.95)
    print(f"\nSelected {len(feature_cols)} features for LSTM after correlation filtering")
    print(f"Features: {', '.join(feature_cols[:10])}...")
    print()

    print("Generating LSTM forecasts (this will take several minutes)...")
    print("Note: LSTM refits every 20 days with early stopping for efficiency")
    print()

    # Generate LSTM forecasts
    lstm_fcst = rolling_lstm_forecast(
        data=features_df,
        target_col='rv',
        feature_cols=feature_cols,
        seq_len=30,  # 30-day lookback
        train_window=756,  # 3 years
        refit_freq=20,  # Monthly refit
        lstm_hidden=64,
        lstm_layers=2,
        epochs=50,  # Max epochs (early stopping typically stops around 20-30)
        batch_size=32,
        verbose=True
    )

    print(f"\n✓ LSTM forecasts: {len(lstm_fcst)} predictions")
    print()

    # =================================================================
    # 5. BASELINE METHODS
    # =================================================================
    print("=" * 80)
    print("4. GENERATING BASELINE FORECASTS")
    print("=" * 80)

    # Historical volatility
    hist_vol = df['returns'].rolling(20).std() * np.sqrt(252)

    # EWMA
    ewma_vol = df['returns'].ewm(halflife=10).std() * np.sqrt(252)

    print("✓ Historical volatility (20-day)")
    print("✓ EWMA (halflife=10)")
    print()

    # =================================================================
    # 6. OUT-OF-SAMPLE EVALUATION
    # =================================================================
    print("=" * 80)
    print("5. OUT-OF-SAMPLE FORECAST EVALUATION")
    print("=" * 80)

    # Get test period data
    test_rv = df.loc[test_start:, 'rv']

    # Create comparison object
    comp = VolatilityComparison(test_rv)
    comp.add_forecast('GARCH(1,1)', garch_fcst)
    comp.add_forecast('EGARCH(1,1)', egarch_fcst)
    comp.add_forecast('LSTM', lstm_fcst)
    comp.add_forecast('Historical Vol', hist_vol.loc[test_start:])
    comp.add_forecast('EWMA', ewma_vol.loc[test_start:])

    # Compute metrics
    metrics = comp.compute_metrics()

    print("\n" + "=" * 80)
    print(metrics[['Model', 'N', 'RMSE', 'MAE', 'QLIKE', 'R²', 'Bias']].to_string(index=False))
    print("=" * 80)

    # Find best models
    best_qlike = metrics.nsmallest(1, 'QLIKE').iloc[0]
    best_r2 = metrics.nlargest(1, 'R²').iloc[0]

    print(f"\n✓ Best QLIKE: {best_qlike['Model']} ({best_qlike['QLIKE']:.4f})")
    print(f"✓ Best R²:    {best_r2['Model']} ({best_r2['R²']:.4f})")
    print()

    # Statistical comparison
    print("=" * 80)
    print("STATISTICAL COMPARISON: GARCH vs LSTM")
    print("=" * 80)

    if 'GARCH(1,1)' in comp.forecasts and 'LSTM' in comp.forecasts:
        dm_result = comp.diebold_mariano_test('GARCH(1,1)', 'LSTM', loss_func='qlike')
        print(f"\nDiebold-Mariano Test (QLIKE loss):")
        print(f"  H0: Equal predictive accuracy")
        print(f"  Test statistic: {dm_result['DM_statistic']:.3f}")
        print(f"  P-value:        {dm_result['p_value']:.4f}")
        print(f"  Result:         {'Significantly different' if dm_result['significant'] else 'Not significantly different'}")
        if dm_result['significant']:
            print(f"  Better model:   {dm_result['better_model']}")
        print()

    # =================================================================
    # 7. VISUALIZATIONS
    # =================================================================
    print("=" * 80)
    print("6. GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Forecast comparison plot
    fig1 = plot_volatility_comparison(
        test_rv,
        {
            'GARCH': garch_fcst.loc[test_start:],
            'EGARCH': egarch_fcst.loc[test_start:],
            'LSTM': lstm_fcst,
            'EWMA': ewma_vol.loc[test_start:]
        },
        title=f"{ticker} Volatility Forecasts: GARCH vs LSTM",
        highlight_periods={'COVID': ('2020-02-01', '2020-05-01')} if test_start <= '2020-02-01' else None
    )
    plt.savefig('comparison_forecasts.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_forecasts.png")
    plt.close()

    # Forecast errors
    fig2 = plot_forecast_errors(
        test_rv,
        {
            'GARCH': garch_fcst,
            'EGARCH': egarch_fcst,
            'LSTM': lstm_fcst
        }
    )
    plt.savefig('comparison_errors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_errors.png")
    plt.close()

    # Scatter plots
    fig3 = plot_scatter_comparison(
        test_rv,
        {
            'GARCH': garch_fcst,
            'EGARCH': egarch_fcst,
            'LSTM': lstm_fcst
        }
    )
    plt.savefig('comparison_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_scatter.png")
    plt.close()

    print()

    # =================================================================
    # 8. ECONOMIC VALUE: VOLATILITY TARGETING
    # =================================================================
    print("=" * 80)
    print("7. EVALUATING ECONOMIC VALUE (VOLATILITY TARGETING)")
    print("=" * 80)

    print("\nRunning backtests...")

    test_returns = df.loc[test_start:, 'returns']

    backtests = {}

    # GARCH strategy
    if len(garch_fcst.loc[test_start:]) > 0:
        garch_weights = vol_target_weights(garch_fcst.loc[test_start:], sigma_star=0.10, w_max=2.0)
        backtests['GARCH'] = run_backtest(test_returns, garch_weights, tc_bps=5, slip_bps=1)

    # EGARCH strategy
    if len(egarch_fcst.loc[test_start:]) > 0:
        egarch_weights = vol_target_weights(egarch_fcst.loc[test_start:], sigma_star=0.10, w_max=2.0)
        backtests['EGARCH'] = run_backtest(test_returns, egarch_weights, tc_bps=5, slip_bps=1)

    # LSTM strategy
    if len(lstm_fcst) > 0:
        lstm_weights = vol_target_weights(lstm_fcst, sigma_star=0.10, w_max=2.0)
        backtests['LSTM'] = run_backtest(test_returns, lstm_weights, tc_bps=5, slip_bps=1)

    # EWMA strategy
    ewma_weights = vol_target_weights(ewma_vol.loc[test_start:], sigma_star=0.10, w_max=2.0)
    backtests['EWMA'] = run_backtest(test_returns, ewma_weights, tc_bps=5, slip_bps=1)

    # Buy & Hold
    bh_weights = pd.Series(1.0, index=test_returns.index)
    backtests['Buy & Hold'] = run_backtest(test_returns, bh_weights, tc_bps=5, slip_bps=1)

    # Results table
    bt_summary = pd.DataFrame({
        'Strategy': [name for name in backtests.keys()],
        'Sharpe Ratio': [bt['sharpe'] for bt in backtests.values()],
        'Max DD (%)': [bt['max_drawdown'] * 100 for bt in backtests.values()],
        'Final Equity': [bt['equity'].iloc[-1] for bt in backtests.values()],
        'Avg Turnover': [bt['turnover'] for bt in backtests.values()]
    })

    print("\n" + "=" * 80)
    print(bt_summary.to_string(index=False))
    print("=" * 80)

    # Find best strategy
    best_sharpe = bt_summary.nlargest(1, 'Sharpe Ratio').iloc[0]
    best_dd = bt_summary.nsmallest(1, 'Max DD (%)').iloc[0]

    print(f"\n✓ Highest Sharpe: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")
    print(f"✓ Lowest DD:      {best_dd['Strategy']} ({best_dd['Max DD (%)']:.2f}%)")
    print()

    # Plot backtest results
    fig4 = plot_backtest_results(backtests)
    plt.savefig('comparison_backtests.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: comparison_backtests.png")
    plt.close()

    # =================================================================
    # 9. FINAL SUMMARY
    # =================================================================
    print("\n" + "=" * 80)
    print("RESEARCH SUMMARY")
    print("=" * 80)

    print(f"\n📊 Data: {ticker} ({start_date} to {end_date})")
    print(f"   Training: {df.loc[:train_end].index[0].date()} to {df.loc[:train_end].index[-1].date()}")
    print(f"   Testing:  {df.loc[test_start:].index[0].date()} to {df.loc[test_start:].index[-1].date()}")

    print(f"\n✓ Stylized Facts:")
    print(f"   Volatility clustering: {'Confirmed' if facts['volatility_clustering'] else 'Not found'}")
    print(f"   Heavy tails: {'Confirmed' if facts['heavy_tails'] else 'Not found'}")

    print(f"\n✓ Forecast Accuracy (Test Period):")
    print(f"   Best QLIKE: {best_qlike['Model']} = {best_qlike['QLIKE']:.4f}")
    print(f"   Best R²:    {best_r2['Model']} = {best_r2['R²']:.4f}")

    print(f"\n✓ Economic Value (Sharpe Ratios):")
    for _, row in bt_summary.iterrows():
        print(f"   {row['Strategy']:15s}: {row['Sharpe Ratio']:>6.2f}")

    print(f"\n✓ Key Findings:")

    # Compare GARCH vs LSTM
    garch_metrics = metrics[metrics['Model'] == 'GARCH(1,1)'].iloc[0] if 'GARCH(1,1)' in metrics['Model'].values else None
    lstm_metrics = metrics[metrics['Model'] == 'LSTM'].iloc[0] if 'LSTM' in metrics['Model'].values else None

    if garch_metrics is not None and lstm_metrics is not None:
        qlike_improvement = (garch_metrics['QLIKE'] - lstm_metrics['QLIKE']) / garch_metrics['QLIKE'] * 100

        if lstm_metrics['QLIKE'] < garch_metrics['QLIKE']:
            print(f"   • LSTM outperforms GARCH by {abs(qlike_improvement):.1f}% (QLIKE)")
        else:
            print(f"   • GARCH outperforms LSTM by {abs(qlike_improvement):.1f}% (QLIKE)")

    # Compare Sharpe ratios
    if 'GARCH' in backtests and 'LSTM' in backtests:
        sharpe_diff = backtests['LSTM']['sharpe'] - backtests['GARCH']['sharpe']
        if sharpe_diff > 0.1:
            print(f"   • LSTM strategy has higher Sharpe (+{sharpe_diff:.2f})")
        elif sharpe_diff < -0.1:
            print(f"   • GARCH strategy has higher Sharpe (+{abs(sharpe_diff):.2f})")
        else:
            print(f"   • GARCH and LSTM strategies have similar Sharpe ratios")

    # Compare to buy & hold
    if 'LSTM' in backtests:
        bh_sharpe = backtests['Buy & Hold']['sharpe']
        lstm_sharpe = backtests['LSTM']['sharpe']
        improvement = lstm_sharpe - bh_sharpe
        print(f"   • Vol-targeting improves Sharpe by {improvement:.2f} vs Buy & Hold")

    # print(f"\n✓ Generated Figures:")
    # print(f"   • comparison_forecasts.png  - Forecast time series")
    # print(f"   • comparison_errors.png     - Forecast errors")
    # print(f"   • comparison_scatter.png    - Actual vs predicted")
    # print(f"   • comparison_backtests.png  - Backtest results")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
