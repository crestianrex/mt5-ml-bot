import pandas as pd

df = pd.read_csv("/home/kyani/Desktop/new/mt5-ml-bot/results/trades_hybrid_adaptive.csv")

# Convert time columns to datetime objects FIRST
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

# Calculate trade duration in minutes FIRST
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

# Analyze SL/TP hits with improved heuristic
sl_hits = 0
tp_hits = 0
other_exits = 0

# Tolerance for price comparison (e.g., 0.1 pip for EURGBP#)
tolerance = 0.00001 # Adjust based on typical price precision

for index, row in df.iterrows():
    if row['direction'] == 'long':
        # Check for SL hit (negative PnL and exit price at or below SL)
        if row['pnl'] <= 0 and (row['exit_price'] <= row['sl'] + tolerance):
            sl_hits += 1
        # Check for TP hit (positive PnL and exit price at or above TP)
        elif row['pnl'] > 0 and (row['exit_price'] >= row['tp'] - tolerance):
            tp_hits += 1
        else:
            other_exits += 1
    elif row['direction'] == 'short':
        # Check for SL hit (negative PnL and exit price at or above SL)
        if row['pnl'] <= 0 and (row['exit_price'] >= row['sl'] - tolerance):
            sl_hits += 1
        # Check for TP hit (positive PnL and exit price at or below TP)
        elif row['pnl'] > 0 and (row['exit_price'] <= row['tp'] + tolerance):
            tp_hits += 1
        else:
            other_exits += 1

print(f"--- Trade Analysis Summary ---")
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
print(f"Trades Closed by Other Means (e.g., force close, breakeven): {other_exits}")