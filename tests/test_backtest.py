import pandas as pd
from src.eval.backtest import vol_target_weights


def test_weight_bounds():
    idx = pd.date_range('2020-01-01', periods=5)
    sig = pd.Series([0.05,0.10,0.20,0.40,0.80], index=idx)
    w = vol_target_weights(sig, sigma_star=0.10, w_max=1.5)
    assert (w>=0).all() and (w<=1.5).all()