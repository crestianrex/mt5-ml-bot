
<!-- WORK FLOW DIAGRAM -->
          ┌───────────────────┐
          │   Historical Data │
          └────────┬──────────┘
                   │
                   ▼
            ┌─────────────┐
            │  train.py   │
            │  (One-time  │
            │  training)  │
            └─────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ Saved Ensembles│
         │  models/*.pkl  │
         └─────┬──────────┘
               │
        ┌──────┴──────────┐
        │                 │
        ▼                 ▼
 ┌───────────────┐   ┌───────────────┐
 │ backtest_hybrid│   │ main_hybrid.py │
 │  (Simulated   │   │  Live trading  │
 │  trading)     │   │  bot)          │
 └─────┬─────────┘   └─────┬─────────┘
       │ Incremental         │ Incremental
       │ retrain every N bars│ retrain every N bars
       ▼                     ▼
 ┌───────────────┐      ┌───────────────┐
 │   Trade Logic │      │  Trade Logic  │
 │ (SimPosition, │      │ (Execution &  │
 │ RiskManager)  │      │ RiskManager)  │
 └─────┬─────────┘      └─────┬─────────┘
       │                        │
       ▼                        ▼
 ┌───────────────┐        ┌───────────────┐
 │ Equity Curve  │        │ Real Trades   │
 │ trades_hybrid │        │   in MT5      │
 │ .csv          │        │               │
 └───────────────┘        └───────────────┘


 <!-- diagram showing the incremental update process per bar -->
           ┌──────────────────────────────┐
          │      New bar arrives         │
          └───────────────┬─────────────┘
                          │
                          ▼
           ┌─────────────────────────┐
           │  Build features (X)     │
           │  & labels (y)           │
           └───────────────┬─────────┘
                           │
                           ▼
           ┌─────────────────────────┐
           │  Merge features & labels│
           └───────────────┬─────────┘
                           │
                           ▼
          ┌─────────────────────────────┐
          │ Check retrain condition:    │
          │  if bar_count % N == 0      │
          └───────────────┬────────────┘
                          │ Yes
                          ▼
             ┌────────────────────────┐
             │ Incrementally retrain  │
             │  ensemble with all     │
             │  data up to current bar│
             └───────────────┬────────┘
                             │
                             ▼
           ┌─────────────────────────┐
           │ Save updated ensemble   │
           │  to disk (*.pkl)        │
           └───────────────┬─────────┘
                           │
                           ▼
           ┌─────────────────────────┐
           │ Predict direction for   │
           │  current bar using X    │
           └───────────────┬─────────┘
                           │
                           ▼
          ┌──────────────────────────────┐
          │ Manage open positions        │
          │  & open new position if allowed │
          └───────────────┬─────────────┘
                          │
                          ▼
          ┌──────────────────────────────┐
          │ Update equity / trade log   │
          │  (simulation or live)       │
          └──────────────────────────────┘


<!-- diagram comparing live hybrid trading vs backtest hybrid -->
          ┌────────────────────────────────────────────┐
          │                 NEW BAR                    │
          └───────────────┬────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │ Build features & labels     │
            └───────────────┬─────────────┘
                            │
                            ▼
          ┌────────────────────────────────────────────┐
          │ Check incremental retrain condition (every N bars) │
          └───────────────┬────────────────────────────┘
                          │
         ┌───────────────┴─────────────────┐
         │                                 │
         ▼                                 ▼
  ┌───────────────────────┐        ┌───────────────────────┐
  │ Live trading           │        │ Backtest simulation    │
  │ (main_hybrid.py)       │        │ (backtest_hybrid.py)  │
  ├───────────────────────┤        ├───────────────────────┤
  │ Predict with ensemble  │        │ Predict with ensemble  │
  │ Execute trades via MT5 │        │ Simulate positions     │
  │ Manage live PnL        │        │ Update equity curve    │
  │ Save updated ensemble  │        │ Save updated ensemble  │
  └───────────────────────┘        └───────────────────────┘
          │                                 │
          ▼                                 ▼
  ┌───────────────────────┐        ┌───────────────────────┐
  │ Trade log / performance│        │ Equity curve / trades │
  │ (real account)         │        │ (simulation)          │
  └───────────────────────┘        └───────────────────────┘


<!-- full workflow diagram -->
                 ┌─────────────────────────────┐
                 │        train.py             │
                 ├─────────────────────────────┤
                 │ - Fetch historical bars     │
                 │ - Build features & labels   │
                 │ - Fit Ensemble models       │
                 │ - Save trained ensembles    │
                 └───────────────┬─────────────┘
                                 │
                 ┌───────────────┴─────────────┐
                 │ Ensemble Models Saved in    │
                 │         /models/            │
                 └───────────────┬─────────────┘
                                 │
             ┌───────────────────┴────────────────────┐
             │                                           │
             ▼                                           ▼
   ┌─────────────────────┐                    ┌─────────────────────┐
   │ main_hybrid.py      │                    │ backtest_hybrid.py  │
   ├─────────────────────┤                    ├─────────────────────┤
   │ Live trading loop   │                    │ Backtest loop       │
   │ ------------------  │                    │ ------------------  │
   │ 1. Fetch latest bar │                    │ 1. Fetch historical bars slice │
   │ 2. Build features   │                    │ 2. Build features   │
   │ 3. Incremental      │                    │ 3. Incremental      │
   │    retrain ensemble │                    │    retrain ensemble │
   │ 4. Predict & trade  │                    │ 4. Predict & simulate trades │
   │ 5. Update live PnL  │                    │ 5. Update simulated equity curve │
   │ 6. Save ensemble    │                    │ 6. Save updated ensemble │
   └─────────────┬───────┘                    └─────────────┬───────┘
                 │                                           │
                 ▼                                           ▼
       ┌─────────────────────┐                    ┌─────────────────────┐
       │ Real account trades  │                    │ Simulated equity &  │
       │ Trade logs / PnL     │                    │ trades CSV files    │
       └─────────────────────┘                    └─────────────────────┘


<!-- Hybrid Incremental Training – Bar-by-Bar Flow -->
Time →  | Bar 1 | Bar 2 | Bar 3 | Bar 4 | Bar 5 | Bar 6 | Bar 7 | ...
--------------------------------------------------------------------------------
Fetch   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
Features|  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
Labels  |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
--------------------------------------------------------------------------------
Predict |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
Decision|  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
Trade   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   |
--------------------------------------------------------------------------------
Retrain |  ❌   |  ❌   |  ✅   |  ❌   |  ❌   |  ✅   |  ❌   |
Save    |  ❌   |  ❌   |  ✅   |  ❌   |  ❌   |  ✅   |  ❌   |



<!-- Hybrid Incremental Model Timeline -->
Time →   Bar 1      Bar 2      Bar 3      Bar 4      Bar 5      Bar 6      Bar 7      ...
--------------------------------------------------------------------------------------------
Fetch     ✅         ✅         ✅         ✅         ✅         ✅         ✅
Features  ✅         ✅         ✅         ✅         ✅         ✅         ✅
Labels    ✅         ✅         ✅         ✅         ✅         ✅         ✅
--------------------------------------------------------------------------------------------
Predict   ✅         ✅         ✅         ✅         ✅         ✅         ✅
Decision  ✅         ✅         ✅         ✅         ✅         ✅         ✅
Trade     ✅         ✅         ✅         ✅         ✅         ✅         ✅
--------------------------------------------------------------------------------------------
Retrain   ❌         ❌         ✅         ❌         ❌         ✅         ❌
Save      ❌         ❌         💾         ❌         ❌         💾         ❌
--------------------------------------------------------------------------------------------
Ensemble State:
          M1          M1         M2         M2         M2         M3         M3


<!-- Hybrid Bot: Live vs Backtest -->
Time →   Bar 1      Bar 2      Bar 3      Bar 4      Bar 5      Bar 6      Bar 7      ...
--------------------------------------------------------------------------------------------
LIVE BOT
Fetch     ✅         ✅         ✅         ✅         ✅         ✅         ✅
Features  ✅         ✅         ✅         ✅         ✅         ✅         ✅
Labels    ✅         ✅         ✅         ✅         ✅         ✅         ✅
Predict   ✅         ✅         ✅         ✅         ✅         ✅         ✅
Trade     ✅         ✅         ✅         ✅         ✅         ✅         ✅
Retrain   ❌         ❌         ✅         ❌         ❌         ✅         ❌
Save      ❌         ❌         💾         ❌         ❌         💾         ❌
Ensemble  M1          M1         M2         M2         M2         M3         M3
--------------------------------------------------------------------------------------------
BACKTEST (Hybrid)
Fetch     ✅         ✅         ✅         ✅         ✅         ✅         ✅
Features  ✅         ✅         ✅         ✅         ✅         ✅         ✅
Labels    ✅         ✅         ✅         ✅         ✅         ✅         ✅
Predict   ✅         ✅         ✅         ✅         ✅         ✅         ✅
Sim Trade ✅         ✅         ✅         ✅         ✅         ✅         ✅
Retrain   ❌         ❌         ✅         ❌         ❌         ✅         ❌
Save      ❌         ❌         💾         ❌         ❌         💾         ❌
Ensemble  M1          M1         M2         M2         M2         M3         M3
Equity ↑  10000      10050      10120      10120      10180      10250      10250





# XM–MT5 AI/ML Trading Bot (Demo)

1. Create and fund an XM **demo** account.
2. Fill `.env` with login/server.
3. Edit `config.yaml` for symbols/timeframe.
4. `pip install -r requirements.txt`
5. `python main.py` (runs a simple live loop on demo).

**Backtest:** see `src/backtest.py` example.

**WARNING:** For demo use only. Markets are risky.