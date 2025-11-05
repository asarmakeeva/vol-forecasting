"""
Visualization utilities for volatility forecasting research
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_volatility_comparison(actual: pd.Series, forecasts: Dict[str, pd.Series],
                               title: str = "Volatility Forecast Comparison",
                               highlight_periods: Optional[Dict[str, Tuple]] = None,
                               figsize: Tuple[int, int] = (15, 6)):
    """
    Plot actual vs forecasted volatility for multiple models

    Parameters:
    -----------
    actual : pd.Series
        Realized volatility
    forecasts : dict
        Dictionary of {model_name: forecast_series}
    title : str
        Plot title
    highlight_periods : dict
        Dict of {label: (start_date, end_date)} to highlight
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual
    ax.plot(actual.index, actual * 100, label='Realized Vol',
           linewidth=2, color='black', alpha=0.8)

    # Plot forecasts
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (name, forecast), color in zip(forecasts.items(), colors):
        aligned = forecast.reindex(actual.index)
        ax.plot(aligned.index, aligned * 100, label=name,
               linewidth=1.5, alpha=0.7, color=color)

    # Highlight periods if provided
    if highlight_periods:
        for label, (start, end) in highlight_periods.items():
            ax.axvspan(start, end, alpha=0.2, color='red', label=label)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel('Volatility (% annualized)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def plot_forecast_errors(actual: pd.Series, forecasts: Dict[str, pd.Series],
                        figsize: Tuple[int, int] = (15, 8)):
    """
    Plot forecast errors for multiple models

    Parameters:
    -----------
    actual : pd.Series
        Realized volatility
    forecasts : dict
        Dictionary of {model_name: forecast_series}
    figsize : tuple
        Figure size
    """
    n_models = len(forecasts)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize, sharex=True)

    if n_models == 1:
        axes = [axes]

    for ax, (name, forecast) in zip(axes, forecasts.items()):
        # Calculate errors
        aligned = pd.DataFrame({
            'actual': actual,
            'forecast': forecast
        }).dropna()

        errors = (aligned['actual'] - aligned['forecast']) * 100

        # Plot errors
        ax.plot(errors.index, errors, linewidth=0.8, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax.fill_between(errors.index, 0, errors, alpha=0.3)

        # Add statistics
        mae = errors.abs().mean()
        bias = errors.mean()
        ax.text(0.02, 0.95, f'MAE: {mae:.3f}%\nBias: {bias:.3f}%',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel(f'{name}\nError (%)', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=11)
    axes[0].set_title('Forecast Errors (Actual - Predicted)', fontweight='bold', fontsize=12)

    plt.tight_layout()
    return fig


def plot_error_distribution(actual: pd.Series, forecasts: Dict[str, pd.Series],
                           figsize: Tuple[int, int] = (14, 10)):
    """
    Plot error distributions for multiple models

    Parameters:
    -----------
    actual : pd.Series
        Realized volatility
    forecasts : dict
        Dictionary of {model_name: forecast_series}
    figsize : tuple
        Figure size
    """
    n_models = len(forecasts)
    fig, axes = plt.subplots(n_models, 2, figsize=figsize)

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for i, (name, forecast) in enumerate(forecasts.items()):
        # Calculate errors
        aligned = pd.DataFrame({
            'actual': actual,
            'forecast': forecast
        }).dropna()

        errors = aligned['actual'] - aligned['forecast']

        # Histogram
        axes[i, 0].hist(errors, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')

        # Add normal distribution overlay
        from scipy import stats
        mu, sigma = errors.mean(), errors.std()
        x = np.linspace(errors.min(), errors.max(), 100)
        axes[i, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

        axes[i, 0].set_title(f'{name}: Error Distribution', fontweight='bold')
        axes[i, 0].set_xlabel('Error')
        axes[i, 0].set_ylabel('Density')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(errors, dist="norm", plot=axes[i, 1])
        axes[i, 1].set_title(f'{name}: Q-Q Plot', fontweight='bold')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_scatter_comparison(actual: pd.Series, forecasts: Dict[str, pd.Series],
                           figsize: Tuple[int, int] = (15, 5)):
    """
    Scatter plots of actual vs predicted for multiple models

    Parameters:
    -----------
    actual : pd.Series
        Realized volatility
    forecasts : dict
        Dictionary of {model_name: forecast_series}
    figsize : tuple
        Figure size
    """
    n_models = len(forecasts)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]

    for ax, (name, forecast) in zip(axes, forecasts.items()):
        # Align data
        aligned = pd.DataFrame({
            'actual': actual * 100,
            'forecast': forecast * 100
        }).dropna()

        # Scatter plot
        ax.scatter(aligned['forecast'], aligned['actual'], alpha=0.5, s=20)

        # 45-degree line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Perfect forecast')

        # Fit line
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(aligned['forecast'], aligned['actual'])
        fit_line = slope * aligned['forecast'] + intercept
        ax.plot(aligned['forecast'], fit_line, 'g-', linewidth=2, alpha=0.7, label='Fitted line')

        # Statistics
        r2 = r_value ** 2
        ax.text(0.05, 0.95, f'R� = {r2:.3f}\nSlope = {slope:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Predicted Volatility (%)', fontsize=10)
        ax.set_ylabel('Actual Volatility (%)', fontsize=10)
        ax.set_title(f'{name}', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_backtest_results(backtest_results: Dict, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot backtest results for multiple strategies

    Parameters:
    -----------
    backtest_results : dict
        Dictionary of {strategy_name: backtest_dict} from run_backtest
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Equity curves
    for name, result in backtest_results.items():
        axes[0].plot(result['equity'].index, result['equity'].values,
                    label=name, linewidth=1.5, alpha=0.8)

    axes[0].set_title('Equity Curves', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Cumulative Return', fontsize=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Drawdowns
    for name, result in backtest_results.items():
        axes[1].plot(result['drawdown'].index, result['drawdown'].values * 100,
                    label=name, linewidth=1.5, alpha=0.8)

    axes[1].set_title('Drawdowns', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Drawdown (%)', fontsize=10)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(result['drawdown'].index,
                         0, result['drawdown'].values * 100,
                         alpha=0.1, color='red')

    # Rolling Sharpe (6-month window)
    for name, result in backtest_results.items():
        returns = result['returns']
        rolling_sharpe = returns.rolling(126).mean() / returns.rolling(126).std() * np.sqrt(252)
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values,
                    label=name, linewidth=1.5, alpha=0.8)

    axes[2].set_title('Rolling Sharpe Ratio (6-month window)', fontweight='bold', fontsize=12)
    axes[2].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_regime_performance(regime_df: pd.DataFrame, metric: str = 'RMSE',
                           figsize: Tuple[int, int] = (12, 6)):
    """
    Plot model performance by regime

    Parameters:
    -----------
    regime_df : pd.DataFrame
        DataFrame with columns: Model, Regime, and metric columns
    metric : str
        Metric to plot (RMSE, MAE, QLIKE, etc.)
    figsize : tuple
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Pivot for grouped bar chart
    pivot_df = regime_df.pivot(index='Regime', columns='Model', values=metric)

    pivot_df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)

    ax.set_title(f'Model Performance by Regime: {metric}', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlabel('Volatility Regime', fontsize=11)
    ax.legend(title='Model', loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_stylized_facts(returns: pd.Series, figsize: Tuple[int, int] = (15, 10)):
    """
    Plot stylized facts of financial returns

    Parameters:
    -----------
    returns : pd.Series
        Asset returns
    figsize : tuple
        Figure size
    """
    from statsmodels.graphics.tsaplots import plot_acf

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1. Return series
    axes[0, 0].plot(returns.index, returns * 100, linewidth=0.5, alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 0].set_title('Returns Time Series', fontweight='bold')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Return distribution
    axes[0, 1].hist(returns.dropna(), bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Overlay normal
    from scipy import stats
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')

    axes[0, 1].set_title('Return Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Return')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Q-Q plot
    stats.probplot(returns.dropna(), dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Q-Q Plot: Normality Test', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. ACF of returns
    plot_acf(returns.dropna(), lags=30, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF: Returns', fontweight='bold')

    # 5. ACF of squared returns (volatility clustering)
    plot_acf(returns.dropna()**2, lags=30, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('ACF: Squared Returns (Volatility Clustering)', fontweight='bold')

    # 6. ACF of absolute returns
    plot_acf(np.abs(returns.dropna()), lags=30, ax=axes[1, 2], alpha=0.05)
    axes[1, 2].set_title('ACF: Absolute Returns', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_garch_diagnostics(std_resid: pd.Series, figsize: Tuple[int, int] = (14, 10)):
    """
    Plot diagnostic tests for GARCH model

    Parameters:
    -----------
    std_resid : pd.Series
        Standardized residuals from GARCH model
    figsize : tuple
        Figure size
    """
    from statsmodels.graphics.tsaplots import plot_acf
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    clean_resid = std_resid.dropna()

    # 1. Standardized residuals over time
    axes[0, 0].plot(clean_resid.index, clean_resid, linewidth=0.5, alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0, 0].axhline(y=2, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
    axes[0, 0].axhline(y=-2, color='red', linestyle=':', linewidth=0.5, alpha=0.5)
    axes[0, 0].set_title('Standardized Residuals', fontweight='bold')
    axes[0, 0].set_ylabel('Std Residuals')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. ACF of standardized residuals
    plot_acf(clean_resid, lags=20, ax=axes[0, 1], alpha=0.05)
    axes[0, 1].set_title('ACF: Standardized Residuals', fontweight='bold')

    # 3. ACF of squared standardized residuals (test for remaining ARCH)
    plot_acf(clean_resid**2, lags=20, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF: Squared Std Residuals (ARCH test)', fontweight='bold')

    # 4. Q-Q plot
    stats.probplot(clean_resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Normality Test', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_figure(fig, filename: str, dpi: int = 300, formats: List[str] = ['png']):
    """
    Save figure in multiple formats

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    dpi : int
        Resolution
    formats : list
        List of formats to save (png, pdf, svg)
    """
    for fmt in formats:
        fig.savefig(f"{filename}.{fmt}", dpi=dpi, bbox_inches='tight', format=fmt)
        print(f"Saved: {filename}.{fmt}")


if __name__ == "__main__":
    print("Volatility Forecasting Visualization Module")
    print("=" * 50)
    print("\nAvailable plotting functions:")
    print("- plot_volatility_comparison: Compare multiple forecasts")
    print("- plot_forecast_errors: Show forecast errors over time")
    print("- plot_error_distribution: Error distribution analysis")
    print("- plot_scatter_comparison: Actual vs predicted scatter")
    print("- plot_backtest_results: Backtest equity and drawdowns")
    print("- plot_regime_performance: Performance by regime")
    print("- plot_stylized_facts: Stylized facts of returns")
    print("- plot_garch_diagnostics: GARCH model diagnostics")