
import pandas as pd
import numpy as np
import os

def analyze_backtest(equity_curve_path, trades_path):
    # --- Load Data ---
    try:
        eq_df = pd.read_csv(equity_curve_path, index_col='time', parse_dates=True)
        trades_df = pd.read_csv(trades_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find result file: {e.filename}")
        return

    # --- Calculate Core Metrics ---
    
    # 1. Total Return
    initial_equity = eq_df['equity'].iloc[0]
    final_equity = eq_df['equity'].iloc[-1]
    total_return_pct = (final_equity / initial_equity) - 1

    # 2. Maximum Drawdown (MDD)
    running_max = eq_df['equity'].cummax()
    drawdown = (eq_df['equity'] - running_max) / running_max
    max_drawdown_pct = drawdown.min()

    # 3. Sharpe Ratio (annualized)
    # Resample to daily returns for a more stable calculation
    daily_equity = eq_df['equity'].resample('D').last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    if len(daily_returns) > 1:
        # Assuming 0 risk-free rate and 252 trading days
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    # --- Trade Statistics ---
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    average_pnl = trades_df['pnl'].mean()
    
    # --- Print Analysis ---
    print("---")
    print("Backtest Performance Analysis (EURGBP#)")
    print("---")
    print(f"Period: {eq_df.index.min().date()} to {eq_df.index.max().date()}")
    print(f"Initial Equity: ${initial_equity:,.2f}")
    print(f"Final Equity:   ${final_equity:,.2f}")
    print(f"Total Return:   {total_return_pct:.2%}")
    print(f"Max Drawdown:   {max_drawdown_pct:.2%}")
    print(f"Sharpe Ratio:   {sharpe_ratio:.2f} (annualized)")
    print("---")
    print(f"Total Trades:   {total_trades}")
    print(f"Win Rate:       {win_rate:.2%}")
    print(f"Average PnL:    ${average_pnl:,.2f}")
    print("--- PnL Stats ---")
    print(trades_df['pnl'].describe())
    print("---")


if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the project root (from 'src')
    project_root = os.path.dirname(script_dir)
    
    # Define paths relative to the project root
    results_dir = os.path.join(project_root, "results")
    equity_curve_file = os.path.join(results_dir, "equity_curve_hybrid_adaptive.csv")
    trades_file = os.path.join(results_dir, "trades_hybrid_adaptive.csv")
    
    analyze_backtest(equity_curve_file, trades_file)
