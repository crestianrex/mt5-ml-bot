# src/data.py
from __future__ import annotations
import pandas as pd
import MetaTrader5 as mt5  # type: ignore
from loguru import logger

# Map human timeframe string to MT5 timeframe constants (best-effort)
TF_MAP = {
    "M1": getattr(mt5, "TIMEFRAME_M1", None),
    "M5": getattr(mt5, "TIMEFRAME_M5", None),
    "M15": getattr(mt5, "TIMEFRAME_M15", None),
    "M30": getattr(mt5, "TIMEFRAME_M30", None),
    "H1": getattr(mt5, "TIMEFRAME_H1", None),
    "H4": getattr(mt5, "TIMEFRAME_H4", None),
    "D1": getattr(mt5, "TIMEFRAME_D1", None),
}


def fetch_bars(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    """
    Fetch `count` bars from MT5 for `symbol` and return a DataFrame with index=time (UTC).
    Returns an empty DataFrame on error or if no data.
    """
    if count is None:
        logger.debug(f"[{symbol}] `count` is None, fetching max history (36000 bars).")
        count = 36000

    tf = TF_MAP.get(str(timeframe).upper())
    if tf is None:
        logger.warning(f"[{symbol}] Unknown timeframe '{timeframe}' — returning empty DataFrame.")
        return pd.DataFrame()

    try:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        if rates is None or len(rates) == 0:
            logger.warning(f"[{symbol}] No bars fetched for timeframe {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        logger.info(f"[{symbol}] Fetched {len(df)} bars from MT5 for timeframe {timeframe}.")
        if "time" not in df.columns:
            logger.warning(f"[{symbol}] fetched data missing 'time' column — returning empty DataFrame")
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"tick_volume": "volume"})
        # keep canonical columns
        return df[["open", "high", "low", "close", "volume"]].copy()
    except Exception as e:
        logger.exception(f"[{symbol}] Error fetching bars: {e}")
        return pd.DataFrame()


def merge_features_labels(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Merge features (X) and labels (y) with raw df. Returns a DataFrame with 'y', 'close', 'high', 'low', 'volume'.
    Drops rows with NaNs and returns empty DataFrame on failure.
    """
    try:
        out = X.copy()
        out["y"] = y.reindex(out.index)
        if "close" in df.columns:
            out["close"] = df["close"].reindex(out.index)
        if "high" in df.columns:
            out["high"] = df["high"].reindex(out.index)
        if "low" in df.columns:
            out["low"] = df["low"].reindex(out.index)
        out["volume"] = df.get("volume", pd.Series(dtype="float")).reindex(out.index)
        out = out.dropna()
        logger.debug(f"Merged features & labels. Final shape: {out.shape}")
        return out
    except Exception as e:
        logger.exception("Error merging features and labels")
        return pd.DataFrame()
