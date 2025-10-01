# strategy_optimizer.py
import optuna
import pandas as pd
import numpy as np
from loguru import logger
import os
import json
import copy # Import copy module for deepcopy
from sklearn.metrics import roc_auc_score
from backtester import HybridBacktester
from src.config import Cfg, RiskCfg # Import Cfg and RiskCfg
from src.utils import setup_logging

# --- Load Configuration ---
# Load the base Cfg object once
base_cfg_obj = Cfg.from_yaml("config.yaml")

# --- Constants ---
N_TRIALS = base_cfg_obj.optuna_n_trials # Use from Cfg object
STUDY_NAME = "strategy_optimization_v2"
STORAGE_PATH = "sqlite:///optuna_results/strategy_optimization_v2.db"
# NEW: Define a directory for strategy parameters
PARAMS_OUTPUT_DIR = "optuna_results/strategy_params"
PARAMS_OUTPUT_FILE = os.path.join(PARAMS_OUTPUT_DIR, "best_strategy_params.json")

def run_backtest_for_trial(trial: optuna.Trial, params):
    """
    Runs a full backtest for a given set of strategy parameters.
    This function is designed to be called by the objective function.
    """
    # Create a deep copy of the base Cfg object for this trial
    trial_cfg_obj = copy.deepcopy(base_cfg_obj)
    
    # --- Temporarily disable safety features for pure optimization ---
    logger.info("Temporarily disabling safety features (drawdown blocks, watchdog) for this optimization trial.")
    trial_cfg_obj.risk.block_on_drawdown = 1.0  # Set to 100% to effectively disable
    if hasattr(trial_cfg_obj, 'watchdog'):
        trial_cfg_obj.watchdog.max_consecutive_losses = 0  # Set to 0 to disable
        trial_cfg_obj.watchdog.cooldown_hours = 0.0

    # Update the risk parameters of the trial_cfg_obj directly
    trial_cfg_obj.risk.atr_multiplier_sl = params["atr_multiplier_sl"]
    trial_cfg_obj.risk.atr_multiplier_tp = params["atr_multiplier_tp"]
    trial_cfg_obj.risk.trailing_atr_mult = params["trailing_atr_mult"]
    trial_cfg_obj.risk.min_prob_long = params["min_prob_long"]
    trial_cfg_obj.risk.min_prob_short = params["min_prob_short"]

    # Instantiate HybridBacktester with the trial-specific Cfg object
    # The backtester will now use its updated internal logic to fetch the correct data
    bt = HybridBacktester(trial_cfg_obj)
    
    # Run backtester, passing the actual trial object for pruning
    pruning_interval = trial_cfg_obj.optuna_pruning_interval # Use configurable pruning interval
    trades_df, eq_df = bt.run(trial=trial, pruning_interval=pruning_interval)

    if eq_df.empty:
        return 0.0

    # Calculate Sharpe Ratio as the objective metric from the equity curve
    returns = eq_df["equity"].pct_change().dropna()
    
    # Avoid division by zero; if std is 0, Sharpe is 0.
    # Annualize Sharpe Ratio based on timeframe
    timeframe_minutes = trial_cfg_obj.timeframe_minutes()
    if timeframe_minutes is None or returns.std() == 0:
        annualization_factor = 0.0 # Set to 0 if std is 0 to avoid division by zero
    else:
        # Assuming 252 trading days in a year, and 24*60 minutes in a day
        annualization_factor = np.sqrt(252 * (24 * 60 / timeframe_minutes))
    sharpe_ratio = returns.mean() / returns.std() * annualization_factor if returns.std() != 0 else 0.0
    
    logger.info(f"Trial completed. Sharpe Ratio: {sharpe_ratio:.4f}")
    return sharpe_ratio

def objective(trial: optuna.Trial):
    """
    The objective function for the Optuna study.
    It suggests hyperparameters and returns the backtest performance.
    """
    params = {
        "atr_multiplier_sl": trial.suggest_float("atr_multiplier_sl", 0.5, 3.0),
        "atr_multiplier_tp": trial.suggest_float("atr_multiplier_tp", 0.5, 5.0),
        "trailing_atr_mult": trial.suggest_float("trailing_atr_mult", 0.5, 3.0),
        "min_prob_long": trial.suggest_float("min_prob_long", 0.51, 0.7),
        "min_prob_short": trial.suggest_float("min_prob_short", 0.51, 0.7),
    }
    
    # It's important to handle potential exceptions during backtesting
    try:
        return run_backtest_for_trial(trial, params)
    except Exception as e:
        logger.error(f"An error occurred during trial {trial.number}: {e}")
        # Prune the trial by returning a very low value
        return -1.0

def main():
    """
    Main function to run the Optuna study.
    """
    setup_logging()
    logger.info(f"Starting Optuna study '{STUDY_NAME}' with {N_TRIALS} trials.")
    logger.info(f"Storage: {STORAGE_PATH}")
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # Run 10 trials fully before starting pruning
            n_warmup_steps=500,   # A trial must complete 500 bars before being pruned
            interval_steps=base_cfg_obj.optuna_pruning_interval # Align with backtester's pruning interval
        )
    )
    
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=base_cfg_obj.n_jobs)
    
    logger.info("Optimization finished.")
    logger.info(f"Best trial number: {study.best_trial.number}")
    logger.info("Best parameters:")
    best_params = study.best_params
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"Best value (Sharpe Ratio): {study.best_value:.4f}")

    # Save the best parameters to a JSON file
    try:
        # Ensure the output directory exists
        os.makedirs(PARAMS_OUTPUT_DIR, exist_ok=True)
        with open(PARAMS_OUTPUT_FILE, 'w') as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Successfully saved best parameters to {PARAMS_OUTPUT_FILE}")
    except IOError as e:
        logger.error(f"Failed to save parameters to {PARAMS_OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    main()
