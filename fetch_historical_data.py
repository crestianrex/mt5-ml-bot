import MetaTrader5 as mt5
import pandas as pd
import os
import yaml
from loguru import logger

# Configure Loguru for file logging
LOG_FILE = "logs/fetch_historical_data.log"
logger.remove() # Remove default handler
logger.add(os.sys.stderr, level="INFO") # Add back console output
logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="INFO") # Add file output

# --- MT5 Timeframe Mapping ---
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

def fetch_and_save_bars(symbol: str, timeframe: str, count: int, save_dir: str):
    """Fetches historical bars from MT5 and saves them to a CSV file."""
    tf = TF_MAP[timeframe]
    file_path = os.path.join(save_dir, f"{symbol.replace('#', '')}_{timeframe}.csv")

    logger.info(f"Fetching {count} bars for {symbol} ({timeframe})...")
    try:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"[{symbol}] No bars fetched for {timeframe}. Skipping save.")
            return

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        
        # Select and rename columns to match original data.py output
        df = df[["open", "high", "low", "close", "tick_volume"]].rename(columns={"tick_volume": "volume"})
        
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(file_path)
        logger.success(f"[{symbol}] Fetched {len(df)} bars and saved to {file_path}")
    except Exception as e:
        logger.error(f"[{symbol}] Error fetching or saving bars: {e}")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Initialize MT5
    if not mt5.initialize():
        logger.error("MetaTrader5 initialize() failed. Is MT5 terminal running?")
        mt5.shutdown()
        exit()

    logger.info("MetaTrader5 initialized successfully.")

    # Load config from config.yaml
    config_path = "config.yaml" # Assuming this script is run from the project root
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"config.yaml not found at {config_path}. Please run this script from the project root.")
        mt5.shutdown()
        exit()

    symbols = cfg.get("symbols", [])
    timeframe = cfg.get("timeframe", "M5")
    history_bars = cfg.get("history_bars", 2000)
    
    if not symbols:
        logger.warning("No symbols found in config.yaml. Exiting.")
        mt5.shutdown()
        exit()

    # Define save directory
    SAVE_DATA_DIR = "data/historical_data"

    # Fetch and save data for each symbol
    for sym in symbols:
        fetch_and_save_bars(sym, timeframe, history_bars, SAVE_DATA_DIR)

    # Shutdown MT5
    mt5.shutdown()
    logger.info("MetaTrader5 shut down.")