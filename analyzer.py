# analyzer.py
import pandas as pd
import sys
import os

def analyze_trades(df: pd.DataFrame, name: str):
    """Analyzes a dataframe of trades and prints a summary."""
    
    if df.empty:
        print(f"--- No trades to analyze for: {name} ---")
        return

    # Convert time columns to datetime objects FIRST
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])

    # Calculate trade duration in minutes
    df['duration_minutes'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60

    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]

    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)
    win_rate = (num_winning_trades / total_trades) * 100 if total_trades > 0 else 0

    avg_pnl = df['pnl'].mean()
    avg_pnl_winning = winning_trades['pnl'].mean() if num_winning_trades > 0 else 0
    avg_pnl_losing = losing_trades['pnl'].mean() if num_losing_trades > 0 else 0

    max_win = winning_trades['pnl'].max() if num_winning_trades > 0 else 0
    max_loss = losing_trades['pnl'].min() if num_losing_trades > 0 else 0

    avg_duration_all = df['duration_minutes'].mean()
    avg_duration_winning = winning_trades['duration_minutes'].mean() if num_winning_trades > 0 else 0
    avg_duration_losing = losing_trades['duration_minutes'].mean() if num_losing_trades > 0 else 0

    # Analyze SL/TP hits
    sl_hits = 0
    tp_hits = 0
    other_exits = 0
    tolerance = 0.00001

    for index, row in df.iterrows():
        if row.get('direction') == 'long':
            if row['pnl'] <= 0 and (row['exit_price'] <= row['sl'] + tolerance):
                sl_hits += 1
            elif row['pnl'] > 0 and (row['exit_price'] >= row['tp'] - tolerance):
                tp_hits += 1
            else:
                other_exits += 1
        elif row.get('direction') == 'short':
            if row['pnl'] <= 0 and (row['exit_price'] >= row['sl'] - tolerance):
                sl_hits += 1
            elif row['pnl'] > 0 and (row['exit_price'] <= row['tp'] + tolerance):
                tp_hits += 1
            else:
                other_exits += 1

    print(f"--- Trade Analysis Summary for: {name} ---")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {num_winning_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {num_losing_trades} ({100 - win_rate:.2f}%)")
    print(f"Average PnL per Trade: {avg_pnl:.2f}")
    print(f"Average PnL for Winning Trades: {avg_pnl_winning:.2f}")
    print(f"Average PnL for Losing Trades: {avg_pnl_losing:.2f}")
    print(f"Maximum Winning Trade PnL: {max_win:.2f}")
    print(f"Maximum Losing Trade PnL: {max_loss:.2f}")
    print(f"Average Trade Duration (minutes): {avg_duration_all:.2f}")
    print(f"  - Winning Trades Avg Duration: {avg_duration_winning:.2f}")
    print(f"  - Losing Trades Avg Duration: {avg_duration_losing:.2f}")
    print(f"Trades Closed by SL: {sl_hits}")
    print(f"Trades Closed by TP: {tp_hits}")
    print(f"Trades Closed by Other Means: {other_exits}\n")


if __name__ == "__main__":
    trade_files = []
    if len(sys.argv) < 2:
        print("No file path provided. Analyzing all 'trades_*.csv' files in 'results/' directory.")
        try:
            results_dir = 'results'
            trade_files = sorted([
                os.path.join(results_dir, f) 
                for f in os.listdir(results_dir) 
                if f.startswith('trades_') and f.endswith('.csv')
            ])
            if not trade_files:
                print("No trade files found in 'results/'.")
                sys.exit(1)
        except FileNotFoundError:
            print("Error: Could not find the 'results/' directory.")
            sys.exit(1)
    else:
        trade_files = sys.argv[1:]

    for filepath in trade_files:
        print(f"============== Analyzing file: {filepath} ==============")
        try:
            main_df = pd.read_csv(filepath)
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.\n")
            continue

        # --- Overall Analysis ---
        analyze_trades(main_df.copy(), f"Overall Portfolio ({os.path.basename(filepath)})")

        # --- Per-Symbol Analysis ---
        symbols = main_df['symbol'].unique()
        if len(symbols) > 1:
            for sym in symbols:
                symbol_df = main_df[main_df['symbol'] == sym]
                analyze_trades(symbol_df.copy(), f"Symbol: {sym} ({os.path.basename(filepath)})")