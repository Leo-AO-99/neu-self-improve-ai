from pathlib import Path
import glob
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
BTC_PATTERN = "BTC-USDT-candlesticks-*.csv"
ETH_PATTERN = "ETH-USDT-candlesticks-*.csv"

def load_okx_candles(data_dir: Path, pattern: str) -> pd.DataFrame:
    """Load and concat all OKX candlestick CSVs matching pattern."""
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {data_dir / pattern}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return out


def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample OHLC to 1h (use close for last close, open=first open, high=max, low=min)."""
    return df.set_index("open_time").resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
    }).dropna(how="all").reset_index()


def merge_btc_eth_1h(data_dir: Path) -> pd.DataFrame:
    """Load BTC and ETH, resample to 1h, merge on open_time."""
    btc = load_okx_candles(data_dir, BTC_PATTERN)
    eth = load_okx_candles(data_dir, ETH_PATTERN)
    btc_1h = resample_to_1h(btc).rename(columns={"close": "btc_close", "open": "btc_open"})
    eth_1h = resample_to_1h(eth).rename(columns={"close": "eth_close", "open": "eth_open"})
    merged = pd.merge(
        btc_1h[["open_time", "btc_close", "btc_open"]],
        eth_1h[["open_time", "eth_close", "eth_open"]],
        on="open_time",
        how="inner",
    )
    return merged

def load_candles():
    merged = merge_btc_eth_1h(DATA_DIR)
    return merged

def compute_spread_zscore(
    merged: pd.DataFrame,
    lookback: int = 24 * 30,  # ~30 days for hourly
) -> pd.Series:
    """Log ratio spread and rolling z-score."""
    # spread = log(ETH/BTC), then z-score
    log_ratio = np.log(merged["eth_close"].astype(float) / merged["btc_close"].astype(float))
    mean_ = log_ratio.rolling(lookback, min_periods=lookback).mean()
    std_ = log_ratio.rolling(lookback, min_periods=lookback).std()
    z = (log_ratio - mean_) / (std_ + 1e-12)
    return z


