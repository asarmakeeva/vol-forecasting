"""
GARCH Volatility Forecasting Research Analysis
==============================================

Comprehensive analysis comparing GARCH models with other volatility forecasting methods.

Author: Research Team
Date: October 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')


class VolatilityComparison:
    """
    Class for comprehensive volatility forecast comparison
    """

    def __init__(self, actual_vol: pd.Series):
        """
        Initialize with actual realized volatility

        Parameters:
        -----------
        actual_vol : pd.Series
            Ground truth realized volatility
        """
        self.actual = actual_vol.dropna()
        self.forecasts = {}

    def add_forecast(self, name: str, forecast: pd.Series):
        """Add a volatility forecast to compare"""
        # Align with actual
        aligned = forecast.reindex(self.actual.index).dropna()
        self.forecasts[name] = aligned

    def compute_metrics(self) -> pd.DataFrame:
        """
        Compute comprehensive forecast evaluation metrics

        Returns:
        --------
        pd.DataFrame
            Table with all metrics for each forecast
        """
        results = []

        for name, forecast in self.forecasts.items():
            # Align data
            data = pd.DataFrame({
                'actual': self.actual,
                'forecast': forecast
            }).dropna()

            if len(data) == 0:
                continue

            y = data['actual'].values
            yhat = data['forecast'].values

            # Loss functions
            rmse = np.sqrt(np.mean((y - yhat)**2))
            mae = np.mean(np.abs(y - yhat))
            mape = np.mean(np.abs((y - yhat) / (y + 1e-8))) * 100

            # QLIKE (Quasi-Likelihood)
            qlike = np.mean(np.log(yhat**2 + 1e-8) + (y**2) / (yhat**2 + 1e-8))

            # R-squared
            r2 = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)

            # Correlation
            corr = np.corrcoef(y, yhat)[0, 1]

            # Mincer-Zarnowitz regression: actual = a + b * forecast
            mz_slope, mz_intercept, mz_r, _, _ = stats.linregress(yhat, y)

            # Bias statistics
            bias = np.mean(yhat - y)
            abs_bias = np.abs(bias)

            # Directional accuracy (for changes)
            if len(y) > 1:
                actual_change = np.sign(np.diff(y))
                forecast_change = np.sign(np.diff(yhat))
                direction_acc = np.mean(actual_change == forecast_change) * 100
            else:
                direction_acc = np.nan

            results.append({
                'Model': name,
                'N': len(data),
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'QLIKE': qlike,
                'R²': r2,
                'Correlation': corr,
                'Bias': bias,
                'Abs_Bias': abs_bias,
                'MZ_Intercept': mz_intercept,
                'MZ_Slope': mz_slope,
                'MZ_R²': mz_r**2,
                'Dir_Accuracy': direction_acc
            })

        return pd.DataFrame(results)

    def diebold_mariano_test(self, model1: str, model2: str,
                            loss_func: str = 'mse') -> Dict:
        """
        Diebold-Mariano test for forecast comparison

        Tests H0: two forecasts have equal predictive accuracy

        Parameters:
        -----------
        model1, model2 : str
            Names of models to compare
        loss_func : str
            Loss function: 'mse', 'mae', 'qlike'

        Returns:
        --------
        dict
            Test statistic and p-value
        """
        if model1 not in self.forecasts or model2 not in self.forecasts:
            raise ValueError("Both models must be in forecasts")

        # Align data
        data = pd.DataFrame({
            'actual': self.actual,
            'f1': self.forecasts[model1],
            'f2': self.forecasts[model2]
        }).dropna()

        y = data['actual'].values
        f1 = data['f1'].values
        f2 = data['f2'].values

        # Calculate loss differential
        if loss_func == 'mse':
            d = (y - f1)**2 - (y - f2)**2
        elif loss_func == 'mae':
            d = np.abs(y - f1) - np.abs(y - f2)
        elif loss_func == 'qlike':
            qlike1 = np.log(f1**2 + 1e-8) + (y**2) / (f1**2 + 1e-8)
            qlike2 = np.log(f2**2 + 1e-8) + (y**2) / (f2**2 + 1e-8)
            d = qlike1 - qlike2
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")

        # DM statistic
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        dm_stat = d_mean / np.sqrt(d_var / len(d))

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

        return {
            'DM_statistic': dm_stat,
            'p_value': p_value,
            'mean_loss_diff': d_mean,
            'significant': p_value < 0.05,
            'better_model': model1 if d_mean < 0 else model2
        }

    def regime_analysis(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Evaluate forecasts by regime

        Parameters:
        -----------
        regimes : pd.Series
            Regime labels aligned with actual volatility

        Returns:
        --------
        pd.DataFrame
            Metrics by regime for each model
        """
        results = []

        for regime in regimes.unique():
            regime_idx = regimes[regimes == regime].index
            regime_actual = self.actual[regime_idx]

            for name, forecast in self.forecasts.items():
                regime_forecast = forecast[regime_idx]

                # Align
                data = pd.DataFrame({
                    'actual': regime_actual,
                    'forecast': regime_forecast
                }).dropna()

                if len(data) < 10:  # Need minimum observations
                    continue

                y = data['actual'].values
                yhat = data['forecast'].values

                rmse = np.sqrt(np.mean((y - yhat)**2))
                mae = np.mean(np.abs(y - yhat))
                qlike = np.mean(np.log(yhat**2 + 1e-8) + (y**2) / (yhat**2 + 1e-8))

                results.append({
                    'Model': name,
                    'Regime': regime,
                    'N': len(data),
                    'RMSE': rmse,
                    'MAE': mae,
                    'QLIKE': qlike
                })

        return pd.DataFrame(results)

    def error_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze forecast errors for each model

        Returns:
        --------
        dict
            Error statistics for each model
        """
        error_stats = {}

        for name, forecast in self.forecasts.items():
            data = pd.DataFrame({
                'actual': self.actual,
                'forecast': forecast
            }).dropna()

            if len(data) == 0:
                continue

            errors = data['actual'] - data['forecast']

            # Error statistics
            stats_dict = {
                'mean': errors.mean(),
                'std': errors.std(),
                'min': errors.min(),
                'max': errors.max(),
                'skewness': stats.skew(errors),
                'kurtosis': stats.kurtosis(errors),
                'normality_pval': stats.jarque_bera(errors)[1]
            }

            # Autocorrelation test
            if len(errors) > 10:
                lb_result = acorr_ljungbox(errors, lags=[10], return_df=True)
                stats_dict['ljungbox_pval'] = lb_result['lb_pvalue'].values[0]

            error_stats[name] = pd.Series(stats_dict)

        return pd.DataFrame(error_stats).T


class GARCHDiagnostics:
    """
    Diagnostic tests for GARCH models
    """

    @staticmethod
    def standardized_residuals_test(std_resid: pd.Series) -> Dict:
        """
        Test if standardized residuals are white noise

        Parameters:
        -----------
        std_resid : pd.Series
            Standardized residuals from GARCH model

        Returns:
        --------
        dict
            Test results
        """
        clean_resid = std_resid.dropna()

        # Ljung-Box test for autocorrelation
        lb_test = acorr_ljungbox(clean_resid, lags=[10, 20], return_df=True)

        # Ljung-Box on squared residuals (remaining ARCH effects)
        lb_sq_test = acorr_ljungbox(clean_resid**2, lags=[10, 20], return_df=True)

        # Normality test
        jb_stat, jb_pval = stats.jarque_bera(clean_resid)

        # Summary
        return {
            'ljungbox_resid_pval': lb_test['lb_pvalue'].values[-1],
            'ljungbox_sq_resid_pval': lb_sq_test['lb_pvalue'].values[-1],
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pval': jb_pval,
            'mean_resid': clean_resid.mean(),
            'std_resid': clean_resid.std(),
            'skewness': stats.skew(clean_resid),
            'kurtosis': stats.kurtosis(clean_resid),
            'no_autocorr': lb_test['lb_pvalue'].values[-1] > 0.05,
            'no_arch': lb_sq_test['lb_pvalue'].values[-1] > 0.05,
            'normal': jb_pval > 0.05
        }

    @staticmethod
    def leverage_effect_test(returns: pd.Series, vol: pd.Series) -> Dict:
        """
        Test for asymmetric volatility (leverage effect)

        Parameters:
        -----------
        returns : pd.Series
            Asset returns
        vol : pd.Series
            Volatility estimates

        Returns:
        --------
        dict
            Test results
        """
        # Align data
        data = pd.DataFrame({
            'returns': returns,
            'vol': vol
        }).dropna()

        # Create negative return indicator
        data['neg_return'] = (data['returns'] < 0).astype(int)
        data['abs_return'] = np.abs(data['returns'])
        data['future_vol'] = data['vol'].shift(-1)

        # Remove NaN
        data = data.dropna()

        # Regression: vol_t+1 = a + b1 * |r_t| + b2 * I(r_t<0) * |r_t|
        X = data[['abs_return', 'neg_return']]
        X['interaction'] = data['abs_return'] * data['neg_return']
        X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)

        y = data['future_vol']

        model = OLS(y, X).fit()

        # Test if interaction term is significant
        leverage_coef = model.params['interaction']
        leverage_pval = model.pvalues['interaction']

        return {
            'leverage_coefficient': leverage_coef,
            'leverage_pvalue': leverage_pval,
            'leverage_significant': leverage_pval < 0.05,
            'leverage_direction': 'negative' if leverage_coef > 0 else 'positive',
            'r_squared': model.rsquared
        }


def stylized_facts_summary(returns: pd.Series) -> Dict:
    """
    Test for stylized facts of financial returns

    Parameters:
    -----------
    returns : pd.Series
        Asset returns

    Returns:
    --------
    dict
        Summary of stylized facts tests
    """
    clean_returns = returns.dropna()

    # Basic statistics
    summary = {
        'mean_return': clean_returns.mean(),
        'volatility': clean_returns.std(),
        'skewness': stats.skew(clean_returns),
        'kurtosis': stats.kurtosis(clean_returns),
        'min_return': clean_returns.min(),
        'max_return': clean_returns.max()
    }

    # Test for normality
    jb_stat, jb_pval = stats.jarque_bera(clean_returns)
    summary['jarque_bera_pval'] = jb_pval
    summary['normal_distribution'] = jb_pval > 0.05

    # Test for autocorrelation in returns
    lb_returns = acorr_ljungbox(clean_returns, lags=[10], return_df=True)
    summary['ljungbox_returns_pval'] = lb_returns['lb_pvalue'].values[0]
    summary['returns_autocorrelated'] = lb_returns['lb_pvalue'].values[0] < 0.05

    # Test for volatility clustering (autocorrelation in squared returns)
    lb_squared = acorr_ljungbox(clean_returns**2, lags=[10], return_df=True)
    summary['ljungbox_squared_pval'] = lb_squared['lb_pvalue'].values[0]
    summary['volatility_clustering'] = lb_squared['lb_pvalue'].values[0] < 0.05

    # Heavy tails (excess kurtosis > 3)
    summary['heavy_tails'] = stats.kurtosis(clean_returns) > 3

    return summary


def format_research_table(df: pd.DataFrame, float_format: str = '.4f') -> str:
    """
    Format DataFrame as publication-ready table

    Parameters:
    -----------
    df : pd.DataFrame
        Results table
    float_format : str
        Format string for floats

    Returns:
    --------
    str
        Formatted table string
    """
    # Create copy to avoid modifying original
    table = df.copy()

    # Format numeric columns
    for col in table.select_dtypes(include=[np.number]).columns:
        table[col] = table[col].apply(lambda x: f"{x:{float_format}}")

    # Create nice string representation
    output = "\n" + "=" * 80 + "\n"
    output += table.to_string(index=False)
    output += "\n" + "=" * 80 + "\n"

    return output


if __name__ == "__main__":
    print("GARCH Analysis Module")
    print("=====================")
    print()
    print("This module provides tools for comprehensive GARCH model analysis:")
    print("- VolatilityComparison: Compare multiple forecasting methods")
    print("- GARCHDiagnostics: Model diagnostic tests")
    print("- stylized_facts_summary: Test stylized facts of returns")
    print()
    print("Example usage:")
    print("  from research.garch_analysis import VolatilityComparison")
    print("  comp = VolatilityComparison(realized_vol)")
    print("  comp.add_forecast('GARCH', garch_forecast)")
    print("  metrics = comp.compute_metrics()")
