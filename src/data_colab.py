from __future__ import annotations
import pandas as pd
import os
from loguru import logger

# This TF_MAP is not strictly needed for data loading from CSV,
# but kept for compatibility if other parts of the code expect it.
TF_MAP = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
}

# Define the directory where historical data CSVs are stored
# This path is relative to the project root (mt5-ml-bot/)
HISTORICAL_DATA_DIR = "data/historical_data"

def fetch_bars(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Fetches historical bars from a local CSV file."""
    # Construct the expected file path
    file_name = f"{symbol.replace('#', '')}_{timeframe}.csv"
    file_path = os.path.join(HISTORICAL_DATA_DIR, file_name)

    logger.info(f"Attempting to load historical data from {file_path}")
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}. Please ensure data is pre-fetched and available.")
            raise FileNotFoundError(f"Historical data CSV not found for {symbol} {timeframe}")

        df = pd.read_csv(file_path, index_col="time", parse_dates=True)
        
        # Ensure the DataFrame has the expected columns and order
        expected_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in expected_cols):
            logger.error(f"Missing expected columns in {file_path}. Expected: {expected_cols}, Found: {df.columns.tolist()}")
            raise ValueError("CSV file has incorrect columns")

        df = df[expected_cols] # Ensure column order

        # Optionally, truncate data if 'count' is less than available bars
        if len(df) > count:
            df = df.tail(count)
            logger.debug(f"Truncated loaded data to {len(df)} bars (requested {count}).")

        logger.debug(f"[{symbol}] Loaded {len(df)} bars from {file_path} for timeframe {timeframe}")
        return df
    except Exception as e:
        logger.exception(f"[{symbol}] Error loading bars from CSV: {e}")
        raise

def merge_features_labels(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Merges features and labels, identical to original src/data.py."""
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
