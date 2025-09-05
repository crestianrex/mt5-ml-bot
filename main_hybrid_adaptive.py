# main_hybrid_adaptive.py
import os
import time
from dotenv import load_dotenv
from loguru import logger

from src.config import Cfg
from src.mt5_client import MT5Client
from src.risk import RiskManager
from src.execution import Execution
from src.utils import setup_logging, get_training_data, load_ensemble, save_ensemble

# --- Initial Setup ---
load_dotenv()
setup_logging()

def run():
    """
    Main function to run the live trading bot loop.
    """
    cfg = Cfg.from_yaml("config.yaml")
    logger.info("=== Starting MT5 ML Bot (Hybrid Adaptive) ===")

    # --- MT5 Connection ---
    mt5c = MT5Client(
        os.getenv("MT5_LOGIN"),
        os.getenv("MT5_PASSWORD"),
        os.getenv("MT5_SERVER"),
        os.getenv("MT5_PATH")
    )
    if not mt5c.connect():
        logger.error("MT5 connection failed. Exiting.")
        return

    # --- Load Ensembles ---
    ens_per_symbol = {sym: load_ensemble(cfg, sym) for sym in cfg.symbols}
    bar_counters = {sym: 0 for sym in cfg.symbols}
    last_bar_time = {sym: None for sym in cfg.symbols}

    # --- Main Trading Loop ---
    try:
        while True:
            for sym in cfg.symbols:
                try:
                    # Use a temporary Cfg to fetch only the latest bars needed
                    temp_cfg = Cfg.from_yaml("config.yaml")
                    temp_cfg.history_bars = 500 # Fetch enough for features, not the whole history
                    
                    data, X, y = get_training_data(temp_cfg, sym)
                    if data.empty:
                        continue
                        
                    latest_bar_time = data.index[-1]

                    if last_bar_time[sym] != latest_bar_time:
                        last_bar_time[sym] = latest_bar_time
                        bar_counters[sym] += 1
                        logger.info(f"[{sym}] New bar detected at {latest_bar_time}")

                        ens = ens_per_symbol[sym]

                        # --- Incremental Retrain ---
                        if bar_counters[sym] % cfg.retrain_every_bars == 0:
                            logger.info(f"[{sym}] Incremental retraining...")
                            # For retraining, get the full history
                            full_data, full_X, full_y = get_training_data(cfg, sym)
                            ens.fit(full_X, full_y)
                            save_ensemble(ens, sym)

                        # --- Execute Trades ---
                        last_features = X.iloc[[-1]]
                        atr = X["atr_14"].iloc[-1]
                        risk = RiskManager(cfg.risk)
                        exe = Execution(ens, risk)
                        
                        exe.manage_trades(sym, atr)
                        result = exe.trade(sym, last_features, atr)
                        
                        if result.ok:
                            logger.info(f"[{sym}] Trade action successful: {result.message}")
                        else:
                            logger.info(f"[{sym}] Trade action skipped: {result.message}")

                except Exception as e:
                    logger.exception(f"An error occurred in the loop for symbol {sym}: {e}")
            
            time.sleep(cfg.timeframe_seconds() or 60) # Wait for the next bar

    except KeyboardInterrupt:
        logger.info("=== Stopping MT5 ML Bot ===")
    finally:
        mt5c.shutdown()
        logger.info("MT5 shutdown complete.")

if __name__ == "__main__":
    run()