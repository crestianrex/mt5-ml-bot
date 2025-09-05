# train_adaptive.py
import MetaTrader5 as mt5
from loguru import logger

from src.config import Cfg
from src.ensemble import Ensemble
from src.utils import get_training_data, load_optuna_params, save_ensemble, setup_logging

# --- Initial Setup ---
setup_logging()  # Set up logging early
if not mt5.initialize():
    logger.error("MT5 initialize() failed. Please check terminal installation.")
    exit(1)

# --- Main Training Execution ---
def main():
    """
    Main function to run the initial training process for each symbol.
    """
    cfg = Cfg.from_yaml("config.yaml")
    
    for sym in cfg.symbols:
        logger.info(f"--- Starting initial training for {sym} ---")
        
        # 1. Get Data: Fetch bars, build features, and create labels
        data, X, y = get_training_data(cfg, sym)
        if data.empty:
            logger.warning(f"[{sym}] No data returned. Skipping training for this symbol.")
            continue

        # 2. Create Ensemble: Create a new ensemble with the best-tuned params
        logger.info(f"[{sym}] Creating a new ensemble model.")
        model_params = load_optuna_params(sym)
        ens = Ensemble(cfg, model_params=model_params)

        # 3. Fit Model: Train the ensemble on the historical data
        logger.info(f"[{sym}] Fitting the ensemble on {len(X)} data points...")
        ens.fit(X, y)

        # 4. Save Model: Save the freshly trained model to disk
        save_ensemble(ens, sym)
        
        logger.info(f"--- Finished initial training for {sym} ---")

if __name__ == "__main__":
    main()
    mt5.shutdown()