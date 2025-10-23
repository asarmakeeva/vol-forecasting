from arch.univariate import ConstantMean, GARCH, EGARCH, Normal
import pandas as pd


def fit_garch(y: pd.Series, kind: str = "garch"):
    am = ConstantMean(y.dropna())
    if kind == "garch":
        am.volatility = GARCH(p=1, o=0, q=1)
    elif kind == "egarch":
        am.volatility = EGARCH(p=1, o=1, q=1)
    am.distribution = Normal()
    res = am.fit(disp="off")
    return res


def one_step_sigma(res) -> pd.Series:
    fcast = res.forecast(horizon=1, reindex=False)
    return fcast.variance.iloc[:, -1].pow(0.5)