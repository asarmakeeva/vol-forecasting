"""
Time series split utilities for walk-forward validation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Iterator
from datetime import datetime


def monthly_walk_forward(
    df: pd.DataFrame,
    start="2016-01",
    train_years=3,
    step_months=1
) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Monthly walk-forward splits (legacy function, kept for compatibility)

    Parameters:
    -----------
    df : pd.DataFrame
        Data with DatetimeIndex
    start : str
        Start period
    train_years : int
        Training window in years
    step_months : int
        Step size in months

    Yields:
    -------
    Tuple[pd.DatetimeIndex, pd.DatetimeIndex]
        Train and test indices
    """
    idx = df.index.unique().sort_values()
    cur = pd.Period(start, freq='M')
    while cur.end_time < idx.max():
        train_end = cur.end_time
        train_start = (train_end - pd.DateOffset(years=train_years))
        test_end = (cur + step_months).end_time
        tr = (idx >= train_start) & (idx <= train_end)
        te = (idx > train_end) & (idx <= test_end)
        yield idx[tr], idx[te]
        cur += step_months


def walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_window: int,
    step_size: int = 1,
    min_train_size: Optional[int] = None
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate walk-forward train/test splits

    Parameters:
    -----------
    dates : pd.DatetimeIndex
        All available dates
    train_window : int
        Training window size (expanding if None)
    step_size : int
        Number of periods to step forward
    min_train_size : int, optional
        Minimum training size for first split

    Returns:
    --------
    List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        List of (train_dates, test_dates) tuples
    """
    splits = []
    n = len(dates)

    start_idx = min_train_size or train_window

    for i in range(start_idx, n, step_size):
        # Train: expanding window or fixed window
        if train_window is None:
            train_idx = dates[:i]
        else:
            train_idx = dates[max(0, i - train_window):i]

        # Test: next period
        if i < n:
            test_idx = dates[i:i + step_size]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

    return splits


def expanding_window_split(
    dates: pd.DatetimeIndex,
    initial_train_size: int,
    step_size: int = 1
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate expanding window splits

    Train window grows over time, always starting from beginning.

    Parameters:
    -----------
    dates : pd.DatetimeIndex
        All available dates
    initial_train_size : int
        Initial training window size
    step_size : int
        Step size for test period

    Returns:
    --------
    List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        List of (train_dates, test_dates) tuples
    """
    splits = []
    n = len(dates)

    for i in range(initial_train_size, n, step_size):
        train_idx = dates[:i]
        test_idx = dates[i:min(i + step_size, n)]

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split by date

    Parameters:
    -----------
    df : pd.DataFrame
        Data with DatetimeIndex
    split_date : str
        Split date (inclusive in test)

    Returns:
    --------
    train : pd.DataFrame
        Training data
    test : pd.DataFrame
        Test data
    """
    train = df.loc[:split_date].iloc[:-1]  # Exclude split_date from train
    test = df.loc[split_date:]

    return train, test