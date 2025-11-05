import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
#import yaml

TRADING_DAYS = 252

def garman_klass(df: pd.DataFrame) -> pd.Series:
    """Daily GK variance proxy (uses OHLC)."""
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    return 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

def realized_vol_from_daily(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Realized volatility proxy when intraday data unavailable

    Uses rolling sum of Garman-Klass estimator.
    """
    return np.sqrt(garman_klass(df).rolling(window).sum() * TRADING_DAYS)

def realized_vol_proxy(df: pd.DataFrame, kind: str = "gk") -> pd.Series:
    if kind.lower() == "gk":
        var = garman_klass(df)
    elif kind.lower() == "cc":  # close-to-close as fallback
        ret = np.log(df["close"]).diff()
        var = ret**2
    else:
        raise ValueError(f"Unknown realized_vol kind: {kind}")
    # annualized sigma (sqrt of summed daily variance * 252)
    return np.sqrt(var.rolling(5, min_periods=3).sum() * TRADING_DAYS)

def add_har_features(s: pd.Series) -> pd.DataFrame:
    """HAR-style lags of realized vol."""
    out = pd.DataFrame(index=s.index)
    out["rv_lag1"] = s.shift(1)
    out["rv_lag5"] = s.rolling(5).mean().shift(1)
    out["rv_lag22"] = s.rolling(22).mean().shift(1)
    return out

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["weekday"] = df.index.weekday
    out["month"] = df.index.month
    return out

def build_features_for_ticker(path: Path, realized_vol: str, seq_len: int) -> Tuple[str, pd.DataFrame]:
    """Read one parquet, build features/target frame for one ticker."""
    ticker = path.stem.upper()
    df = pd.read_parquet(path)  # expects columns: open, high, low, close, volume
    df = df.sort_index()
    # sanity: enforce lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # returns
    df["ret"] = df["close"].pct_change()
    df["logret"] = np.log(df["close"]).diff()

    # realized volatility target (proxy)
    df["rv"] = realized_vol_proxy(df, realized_vol)

    # HAR features on rv
    har = add_har_features(df["rv"])

    # simple rolling stats on returns
    feats_ret = pd.DataFrame(index=df.index)
    feats_ret["ret_1d"] = df["ret"].shift(1)
    feats_ret["ret_5d"] = df["ret"].rolling(5).mean().shift(1)
    feats_ret["ret_22d"] = df["ret"].rolling(22).mean().shift(1)
    feats_ret["vol_22d"] = df["ret"].rolling(22).std().shift(1) * np.sqrt(TRADING_DAYS)

    # calendar
    cal = add_calendar(df)

    # assemble
    fr = pd.concat(
        [
            df[["open", "high", "low", "close", "volume"]],
            df[["ret", "logret"]],
            df[["rv"]].rename(columns={"rv": "rv_target"}),
            har,
            feats_ret,
            cal,
        ],
        axis=1,
    )

    # drop rows that don't have enough history
    fr = fr.dropna()

    # add ticker column
    fr["ticker"] = ticker
    return ticker, fr

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    # allow CLI override if no config present
    ap.add_argument("--input_path", type=str, default=None)
    ap.add_argument("--output_path", type=str, default=None)
    ap.add_argument("--realized_vol", type=str, default=None)  # gk or cc
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        cfg = load_config(args.config)

    input_path = args.input_path or cfg.get("input_path", "data/raw/daily")
    output_path = args.output_path or cfg.get("output_path", "data/processed")
    realized_vol = (args.realized_vol or cfg.get("realized_vol", "gk")).lower()
    seq_len = args.seq_len or int(cfg.get("seq_len", 30))
    tickers_cfg = cfg.get("tickers", None)
    tickers = [t.upper() for t in (args.tickers or tickers_cfg or [])]

    input_dir = Path(input_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # select files
    files = sorted(glob.glob(str(input_dir / "*.parquet")))
    if tickers:
        files = [str(input_dir / f"{t}.parquet") for t in tickers if (input_dir / f"{t}.parquet").exists()]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")

    per_ticker_paths = []
    merged_frames = []

    for f in files:
        tkr, fr = build_features_for_ticker(Path(f), realized_vol, seq_len)
        # save per-ticker features
        out_file = out_dir / f"{tkr}_features.parquet"
        fr.to_parquet(out_file)
        per_ticker_paths.append(out_file)
        merged_frames.append(fr.assign(ticker=tkr))
        print(f"[OK] {tkr}: {fr.shape[0]} rows → {out_file}")

    # merged multi-ticker panel (optional but handy)
    all_df = pd.concat(merged_frames, axis=0).sort_index()
    all_out = out_dir / "all_tickers_features.parquet"
    all_df.to_parquet(all_out)
    print(f"[OK] Merged: {all_df.shape[0]} rows → {all_out}")

if __name__ == "__main__":
    main()
