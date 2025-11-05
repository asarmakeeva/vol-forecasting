import argparse, pathlib
import pandas as pd
import yfinance as yf

def fetch_daily(ticker: str, start: str) -> pd.DataFrame:
    # group_by='column' forces single-level names when possible
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol or start date.")

    # Normalize columns for both single- and multi-index cases
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]).lower() for col in df.columns]  # ('SPY','Open')->'open'
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Keep consistent column set if Adj Close sneaks in
    rename = {"adj close": "close"} if "close" not in df.columns and "adj close" in df.columns else {}
    df = df.rename(columns=rename)

    df.index = pd.to_datetime(df.index, utc=False)
    df.index.name = "date"
    df = df[["open", "high", "low", "close", "volume"]].dropna(how="all", axis=1)
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", default=["SPY","QQQ","AAPL","MSFT","NVDA"])
    p.add_argument("--start", default="2015-01-01")
    args = p.parse_args()

    out = pathlib.Path("data/raw/daily")
    out.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        df = fetch_daily(t, args.start)
        df.to_parquet(out / f"{t}.parquet")
        print(f"Saved {t}: {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")
