from arch.univariate import ConstantMean, GARCH, EGARCH, Normal, StudentsT, SkewStudent
import pandas as pd
import numpy as np


def fit_garch(y: pd.Series, kind: str = "garch", distribution: str = "normal", p: int = 1, q: int = 1):
    """
    Fit GARCH-family models with flexible specifications

    Parameters:
    -----------
    y : pd.Series
        Time series of returns
    kind : str
        Model type: 'garch', 'egarch', 'gjr'
    distribution : str
        Error distribution: 'normal', 'studentst', 'skewstudent'
    p : int
        GARCH lag order (default: 1)
    q : int
        ARCH lag order (default: 1)

    Returns:
    --------
    ARCHModelResult
        Fitted model object
    """
    am = ConstantMean(y.dropna())

    # Set volatility model
    if kind == "garch":
        am.volatility = GARCH(p=p, o=0, q=q)
    elif kind == "egarch":
        am.volatility = EGARCH(p=p, o=1, q=q)
    elif kind == "gjr":
        am.volatility = GARCH(p=p, o=1, q=q)  # GJR-GARCH has asymmetric term
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    # Set distribution
    if distribution == "normal":
        am.distribution = Normal()
    elif distribution == "studentst":
        am.distribution = StudentsT()
    elif distribution == "skewstudent":
        am.distribution = SkewStudent()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    res = am.fit(disp="off", show_warning=False)
    return res


def one_step_sigma(res) -> pd.Series:
    """Get 1-step ahead volatility forecast"""
    fcast = res.forecast(horizon=1, reindex=False)
    return fcast.variance.iloc[:, -1].pow(0.5)


def multi_step_sigma(res, horizon: int = 5) -> pd.DataFrame:
    """Get multi-step ahead volatility forecasts"""
    fcast = res.forecast(horizon=horizon, reindex=False)
    return fcast.variance.pow(0.5)


def get_model_params(res) -> dict:
    """Extract and interpret GARCH model parameters"""
    params = {}

    # Common parameters
    for param_name in res.params.index:
        params[param_name] = res.params[param_name]

    # Calculate derived statistics
    if 'alpha[1]' in params and 'beta[1]' in params:
        params['persistence'] = params['alpha[1]'] + params['beta[1]']

        if params['persistence'] < 1.0:
            params['half_life'] = np.log(0.5) / np.log(params['persistence'])
            params['unconditional_vol'] = np.sqrt(params['omega'] / (1 - params['persistence']))
        else:
            params['half_life'] = np.inf
            params['unconditional_vol'] = np.nan

    # Extract leverage parameter if present (EGARCH/GJR)
    if 'gamma[1]' in params:
        params['leverage_effect'] = params['gamma[1]']

    return params


def rolling_garch_forecast(returns: pd.Series, window: int = 252, kind: str = "garch",
                          refit_freq: int = 20) -> pd.Series:
    """
    Generate rolling GARCH forecasts with periodic refitting

    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    window : int
        Minimum training window size (default: 252 = 1 year)
    kind : str
        GARCH model type
    refit_freq : int
        Refit model every N days (default: 20 = monthly)

    Returns:
    --------
    pd.Series
        Rolling 1-day ahead volatility forecasts
    """
    forecasts = []
    dates = []

    # Ensure we have enough data
    if len(returns) < window:
        raise ValueError(f"Insufficient data: need at least {window} observations")

    last_fit = None

    for i in range(window, len(returns)):
        # Get training data
        train = returns.iloc[max(0, i-window):i]

        # Refit model periodically or on first iteration
        if last_fit is None or i % refit_freq == 0:
            try:
                model = fit_garch(train * 100, kind=kind)  # scale for numerical stability
                last_fit = model
            except Exception as e:
                # If fit fails, use previous model or skip
                if last_fit is None:
                    continue
                model = last_fit
        else:
            model = last_fit

        # Generate 1-step forecast
        try:
            fcast = model.forecast(horizon=1, reindex=False)
            vol_forecast = np.sqrt(fcast.variance.values[-1, -1]) / 100  # scale back

            forecasts.append(vol_forecast)
            dates.append(returns.index[i])
        except Exception:
            continue

    return pd.Series(forecasts, index=dates, name=f'{kind}_forecast')


def compare_garch_models(returns: pd.Series, models: list = None) -> pd.DataFrame:
    """
    Compare different GARCH specifications using information criteria

    Parameters:
    -----------
    returns : pd.Series
        Time series of returns
    models : list
        List of (kind, distribution) tuples to compare

    Returns:
    --------
    pd.DataFrame
        Comparison table with AIC, BIC, log-likelihood
    """
    if models is None:
        models = [
            ('garch', 'normal'),
            ('garch', 'studentst'),
            ('egarch', 'normal'),
            ('egarch', 'studentst'),
            ('gjr', 'normal'),
            ('gjr', 'studentst'),
        ]

    results = []

    for kind, dist in models:
        try:
            model = fit_garch(returns * 100, kind=kind, distribution=dist)

            results.append({
                'Model': f'{kind.upper()}',
                'Distribution': dist,
                'LogLik': model.loglikelihood,
                'AIC': model.aic,
                'BIC': model.bic,
                'Params': len(model.params)
            })
        except Exception as e:
            print(f"Failed to fit {kind} with {dist}: {e}")
            continue

    df = pd.DataFrame(results)

    # Rank models (lower AIC/BIC is better)
    if len(df) > 0:
        df['AIC_Rank'] = df['AIC'].rank()
        df['BIC_Rank'] = df['BIC'].rank()
        df = df.sort_values('AIC')

    return df