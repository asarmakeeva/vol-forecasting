import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from typing import List, Optional


TRADING_DAYS = 252

def garman_klass(df: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass volatility estimator

    Uses OHLC data to estimate volatility more efficiently than close-to-close.
    """
    log_hl = np.log(df['high']/df['low'])
    log_co = np.log(df['close']/df['open'])
    return 0.5*log_hl**2 - (2*np.log(2)-1)*log_co**2

    per_ticker_paths = []
    merged_frames = []

def realized_vol_from_daily(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Realized volatility proxy when intraday data unavailable

    Uses rolling sum of Garman-Klass estimator.
    """
    return np.sqrt(garman_klass(df).rolling(window).sum() * TRADING_DAYS)

    # merged multi-ticker panel (optional but handy)
    all_df = pd.concat(merged_frames, axis=0).sort_index()
    all_out = out_dir / "all_tickers_features.parquet"
    all_df.to_parquet(all_out)
    print(f"[OK] Merged: {all_df.shape[0]} rows → {all_out}")

def make_supervised(fr: pd.DataFrame, horizon: int = 1, seq_len: int = 30):
    """
    Create supervised learning dataset for DL models

    Parameters:
    -----------
    fr : pd.DataFrame
        Feature DataFrame with 'rv' column
    horizon : int
        Forecast horizon
    seq_len : int
        Sequence length for LSTM

    Returns:
    --------
    X : pd.DataFrame
        Features
    y : pd.Series
        Targets (shifted realized volatility)
    """
    y = fr['rv'].shift(-horizon)
    X_cols = [c for c in fr.columns if c != 'rv']
    return fr[X_cols], y


def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive features for volatility forecasting

    Features include:
    - Returns at multiple horizons
    - Rolling volatilities
    - Volume-based features
    - Momentum indicators
    - Garman-Klass volatility

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data

    Returns:
    --------
    pd.DataFrame
        Feature DataFrame
    """
    features = pd.DataFrame(index=df.index)

    # Returns
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Lagged returns (using important lags from feature analysis)
    # Research shows lags [1, 2, 6, 11, 16] are most predictive
    for lag in [1, 2, 6, 11, 16]:
        features[f'ret_lag_{lag}'] = features['returns'].shift(lag)

    # Squared and absolute returns (volatility proxies)
    features['returns_sq'] = features['returns'] ** 2
    features['returns_abs'] = features['returns'].abs()

    # Rolling volatility at different windows
    for window in [5, 10, 20, 60]:
        features[f'vol_{window}d'] = features['returns'].rolling(window).std() * np.sqrt(TRADING_DAYS)

    # Garman-Klass volatility
    features['gk_vol'] = np.sqrt(garman_klass(df) * TRADING_DAYS)

    # Rolling Garman-Klass
    for window in [5, 10, 20]:
        features[f'gk_vol_{window}d'] = np.sqrt(
            garman_klass(df).rolling(window).mean() * TRADING_DAYS
        )

    # Parkinson volatility (high-low range)
    features['parkinson'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low'])) ** 2 * TRADING_DAYS
    )

    # Volume features (if available)
    if 'volume' in df.columns:
        features['volume'] = df['volume']
        features['volume_chg'] = df['volume'].pct_change()

        # Volume-weighted metrics
        features['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['price_to_vwap'] = df['close'] / features['vwap']

        # Normalized volume
        features['volume_norm'] = df['volume'] / df['volume'].rolling(60).mean()

    # Price-based features
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['close_open_chg'] = (df['close'] - df['open']) / df['open']

    # Momentum indicators
    features['rsi_14'] = compute_rsi(df['close'], window=14)

    # Moving averages
    for window in [10, 20, 50]:
        ma = df['close'].rolling(window).mean()
        features[f'price_to_ma{window}'] = df['close'] / ma

    # Trend indicators
    features['trend_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # EWMA volatility
    for halflife in [5, 10, 20]:
        features[f'ewma_vol_{halflife}'] = (
            features['returns'].ewm(halflife=halflife).std() * np.sqrt(TRADING_DAYS)
        )

    # Realized volatility (target)
    features['rv'] = realized_vol_from_daily(df)

    return features.dropna()


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index

    Parameters:
    -----------
    prices : pd.Series
        Price series
    window : int
        RSI window

    Returns:
    --------
    pd.Series
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def select_lstm_features(features_df: pd.DataFrame,
                        exclude_highly_correlated: bool = True,
                        correlation_threshold: float = 0.95) -> List[str]:
    """
    Select relevant features for LSTM model

    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature DataFrame
    exclude_highly_correlated : bool
        Remove highly correlated features
    correlation_threshold : float
        Correlation threshold for removal

    Returns:
    --------
    list
        Selected feature names
    """
    # Exclude target and NaN-heavy columns
    exclude_cols = ['rv']
    candidate_features = [c for c in features_df.columns if c not in exclude_cols]

    # Remove columns with too many NaNs
    valid_features = []
    for col in candidate_features:
        if features_df[col].isna().sum() / len(features_df) < 0.1:
            valid_features.append(col)

    if not exclude_highly_correlated:
        return valid_features

    # Remove highly correlated features
    corr_matrix = features_df[valid_features].corr().abs()

    # Find pairs of highly correlated features
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    for column in upper_tri.columns:
        if any(upper_tri[column] > correlation_threshold):
            to_drop.append(column)

    selected = [f for f in valid_features if f not in to_drop]

    return selected


def normalize_features(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      method: str = 'standardize') -> tuple:
    """
    Normalize features using training statistics

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training features
    test_df : pd.DataFrame
        Test features
    method : str
        'standardize' (z-score) or 'minmax'

    Returns:
    --------
    train_norm : pd.DataFrame
        Normalized training features
    test_norm : pd.DataFrame
        Normalized test features
    scaler_params : dict
        Scaling parameters
    """
    if method == 'standardize':
        mean = train_df.mean()
        std = train_df.std()

        train_norm = (train_df - mean) / (std + 1e-8)
        test_norm = (test_df - mean) / (std + 1e-8)

        scaler_params = {'mean': mean, 'std': std, 'method': 'standardize'}

    elif method == 'minmax':
        min_val = train_df.min()
        max_val = train_df.max()

        train_norm = (train_df - min_val) / (max_val - min_val + 1e-8)
        test_norm = (test_df - min_val) / (max_val - min_val + 1e-8)

        scaler_params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return train_norm, test_norm, scaler_params


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("- garman_klass: Garman-Klass volatility estimator")
    print("- realized_vol_from_daily: Realized vol proxy")
    print("- create_volatility_features: Comprehensive feature creation")
    print("- select_lstm_features: Feature selection for LSTM")
    print("- normalize_features: Feature normalization")
