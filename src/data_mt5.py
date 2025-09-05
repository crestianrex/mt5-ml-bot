import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime, timedelta

# Map timeframe strings to MT5 constants
timeframe_map = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
}

def fetch_mt5_data(symbol="XAUUSD", timeframe="M5", n_bars=5000) -> pd.DataFrame:
    """
    Fetch OHLCV data from MT5 for the given symbol and timeframe.
    """
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")


    tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
    utc_from = datetime.now(pytz.utc) - timedelta(minutes=n_bars * 5)
    rates = mt5.copy_rates_from(symbol, tf, utc_from, n_bars)


    if rates is None:
        raise RuntimeError(f"Failed to fetch data for {symbol}")


    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    return df[["open", "high", "low", "close", "tick_volume"]].rename(columns={"tick_volume": "volume"})