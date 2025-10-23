import pandas as pd
from src.data.features import garman_klass


def test_gk_positive():
    df = pd.DataFrame({
    'open':[100,101], 'high':[105,103], 'low':[99,100], 'close':[102,102]
    })
    s = garman_klass(df)
    assert s.notna().all()
