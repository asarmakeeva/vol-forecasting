from typing import Iterator, Tuple
import pandas as pd


def monthly_walk_forward(df: pd.DataFrame, start="2016-01", train_years=3, step_months=1) -> Iterator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
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