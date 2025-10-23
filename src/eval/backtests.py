import numpy as np, pandas as pd


def vol_target_weights(sig_forecast: pd.Series, sigma_star=0.10, w_max=2.0, sigma_min=1e-4, long_only=True):
    w = sigma_star / np.maximum(sig_forecast, sigma_min)
    if long_only:
        w = np.clip(w, 0.0, w_max)
    else:
        w = np.clip(w, -w_max, w_max)
    return pd.Series(w, index=sig_forecast.index)


def run_backtest(returns: pd.Series, w: pd.Series, tc_bps=5, slip_bps=1):
    w = w.reindex(returns.index).ffill().fillna(0.0)
    ret_gross = w.shift(1) * returns # use yesterday’s weight
    turnover = w.diff().abs().fillna(0.0)
    costs = turnover * (tc_bps + slip_bps) / 1e4
    ret_net = ret_gross - costs
    eq = (1 + ret_net).cumprod()
    dd = eq / eq.cummax() - 1
    ann = 252
    sharpe = np.sqrt(ann) * ret_net.mean() / (ret_net.std() + 1e-12)
    mdd = dd.min()
    return {
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "turnover": turnover.mean(),
        "equity": eq,
        "drawdown": dd,
        "returns": ret_net,
        }