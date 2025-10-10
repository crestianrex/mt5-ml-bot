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
N_TRIALS = 1
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
    # These are now handled by RiskController within HybridBacktester
    # trial_cfg_obj.risk.atr_multiplier_sl = params["atr_multiplier_sl"]
    # trial_cfg_obj.risk.atr_multiplier_tp = params["atr_multiplier_tp"]
    # trial_cfg_obj.risk.trailing_atr_mult = params["trailing_atr_mult"]
    # trial_cfg_obj.risk.min_prob_long = params["min_prob_long"]
    # trial_cfg_obj.risk.min_prob_short = params["min_prob_short"]

    # Instantiate HybridBacktester with the trial-specific Cfg object
    # The backtester will now use its updated internal logic to fetch the correct data
    bt = HybridBacktester(trial_cfg_obj)
    
    # Run backtester, passing the actual trial object for pruning
    pruning_interval = trial_cfg_obj.optuna_pruning_interval # Use configurable pruning interval
    trades_df, eq_df = bt.run(trial=trial, pruning_interval=pruning_interval)

    # Save the state of the risk controller for this specific trial
    # This allows us to retrieve the state of the best trial later
    try:
        # Ensure the directory for temporary state files exists
        os.makedirs(PARAMS_OUTPUT_DIR, exist_ok=True)
        trial_state_file = os.path.join(PARAMS_OUTPUT_DIR, f"ts_state_trial_{trial.number}.json")
        bt.risk_controller.state_file = trial_state_file
        bt.risk_controller.save_state()
        logger.info(f"Saved state for trial {trial.number} to {trial_state_file}")
    except Exception as e:
        logger.error(f"Failed to save state for trial {trial.number}: {e}")

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
    It now runs the backtest with RiskController enabled and returns the performance.
    """
    # No longer suggesting risk parameters, as they are handled by RiskController
    # Optuna can still be used to optimize other parameters if needed, or just run a single backtest.
    params = {} # Empty params, as RiskController handles risk parameters
    
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
    # The best parameters from Optuna are no longer directly risk parameters
    # They might be other parameters if Optuna is used for meta-optimization
    # For now, we focus on the RiskController's learned state.
    logger.info(f"Best value (Sharpe Ratio): {study.best_value:.4f}")

    # --- Save the state of the best trial's RiskController ---
    try:
        import shutil
        import glob

        best_trial_num = study.best_trial.number
        best_trial_state_file = os.path.join(PARAMS_OUTPUT_DIR, f"ts_state_trial_{best_trial_num}.json")
        final_state_file = base_cfg_obj.thompson_sampling.state_file

        # Ensure the destination directory exists
        final_state_dir = os.path.dirname(final_state_file)
        if final_state_dir:
            os.makedirs(final_state_dir, exist_ok=True)

        shutil.copy(best_trial_state_file, final_state_file)
        logger.info(f"Saved best RiskController state from trial {best_trial_num} to {final_state_file}")

        # Clean up all temporary trial state files
        for f in glob.glob(os.path.join(PARAMS_OUTPUT_DIR, "ts_state_trial_*.json")):
            os.remove(f)
        logger.info("Cleaned up temporary trial state files.")

    except FileNotFoundError:
        logger.error(f"Could not find state file for best trial: {best_trial_state_file}")
    except Exception as e:
        logger.error(f"An error occurred while saving the best trial's state: {e}")

if __name__ == "__main__":
    main()
