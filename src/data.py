### `src/data.py`

from __future__ import annotations
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from loguru import logger

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

def fetch_bars(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    tf = TF_MAP[timeframe]
    try:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"[{symbol}] No bars fetched for {timeframe}")
            raise RuntimeError(f"No rates for {symbol} {timeframe}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        logger.debug(f"[{symbol}] Fetched {len(df)} bars for timeframe {timeframe}")
        return df[["open","high","low","close","tick_volume"]].rename(columns={"tick_volume":"volume"})
    except Exception as e:
        logger.exception(f"[{symbol}] Error fetching bars: {e}")
        raise

def merge_features_labels(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    try:
        out = X.copy()
        out["y"] = y
        out["close"] = df["close"]
        out["high"] = df["high"]
        out["low"] = df["low"]
        out["volume"] = df.get("volume")
        out = out.dropna()
        logger.debug(f"Merged features & labels. Final shape: {out.shape}")
        return out
    except Exception as e:
        logger.exception("Error merging features and labels")
        raise
