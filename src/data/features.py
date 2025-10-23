import numpy as np
import pandas as pd


TRADING_DAYS = 252


def garman_klass(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df['high']/df['low'])
    log_co = np.log(df['close']/df['open'])
    return 0.5*log_hl**2 - (2*np.log(2)-1)*log_co**2


def realized_vol_from_daily(df: pd.DataFrame) -> pd.Series:
# proxy when intraday unavailable
    return np.sqrt(garman_klass(df).rolling(5).sum() * TRADING_DAYS)


def make_supervised(fr: pd.DataFrame, horizon: int = 1, seq_len: int = 30):
    # create targets σ_{t+1} and sequences for DL
    y = fr['rv'].shift(-horizon)
    X_cols = [c for c in fr.columns if c != 'rv']
    # DL sequences: build rolling windows [t-seq_len+1, t]
    return fr[X_cols], y