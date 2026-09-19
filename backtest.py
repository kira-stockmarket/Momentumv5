import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from train_tier1_lgbm import engineer_features
import warnings

warnings.filterwarnings("ignore")

def calculate_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1.0
    return drawdown.min()

def run_backtest():
    print("Loading data and models for backtest...")
    raw_df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(raw_df)
    
    lgbm = joblib.load("tier1_lgbm_model.pkl")
    thresh = joblib.load("tier1_threshold.pkl")
    ppo = PPO.load("tier2_ppo_agent.zip")
    
    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    
    # Restrict backtest to the last 2 years for Out-of-Sample evaluation
    data["Date"] = pd.to_datetime(data["Date"])
    start_date = data["Date"].max() - pd.Timedelta(days=730)
    bt_data = data[data["Date"] >= start_date].copy()
    
    print(f"Generating Tier 1 Signals from {start_date.date()} to {bt_data['Date'].max().date()}...")
    bt_data["Prob"] = lgbm.predict_proba(bt_data[feature_cols])[:, 1]
    bt_data["Signal"] = (bt_data["Prob"] >= thresh).astype(int)
    
    signals = bt_data[bt_data["Signal"] == 1].sort_values("Date")
    
    # ---------------------------------------------------------
    # STEP 1: Pre-calculate all potential RL trade outcomes
    # ---------------------------------------------------------
    print(f"Pre-calculating RL exit strategies for {len(signals)} raw signals...")
    ticker_groups = dict(tuple(bt_data.groupby("Ticker")))
    potential_trades = []
    
    for _, sig in signals.iterrows():
        ticker = sig["Ticker"]
        entry_date = sig["Date"]
        
        t_data = ticker_groups[ticker]
        trade_slice = t_data[t_data["Date"] >= entry_date].head(16).reset_index(drop=True)
        
        if len(trade_slice) < 2:
            continue
            
        entry_price = trade_slice.iloc[0]["Close"]
        fee = 0.001 
        
        for step_idx in range(1, len(trade_slice)):
            row = trade_slice.iloc[step_idx]
            unrealized_pnl = (row["Close"] - entry_price) / entry_price
            
            if unrealized_pnl <= -0.06:  # Hard stop safety guard
                break
                
            obs = np.array([
                unrealized_pnl,
                min(1.0, step_idx / 15.0),
                row["ATR_14"],
                row["ROC_20"],
                row["Dist_to_High"]
            ], dtype=np.float32)
            
            action, _ = ppo.predict(obs, deterministic=True)
            if action == 1: 
                break
                
        realized_return = ((row["Close"] - entry_price) / entry_price) - fee
        potential_trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": row["Date"],
            "Ticker": ticker,
            "Return": realized_return,
            "Bars_Held": step_idx
        })

    trades_df = pd.DataFrame(potential_trades).sort_values("Entry_Date")

    # ---------------------------------------------------------
    # STEP 2: Chronological Portfolio & Capital Simulator
    # ---------------------------------------------------------
    print("Running chronological portfolio simulation with strict capital constraints...")
    
    starting_capital = 1_000_000.0
    cash = starting_capital
    max_positions = 20
    allocation_per_trade = 0.05 # 5% per trade
    
    active_positions = []
    executed_trades = []
    equity_history = []
    
    all_dates = sorted(bt_data["Date"].unique())
    trade_idx = 0
    num_potential = len(trades_df)
    
    for current_date in all_dates:
        # 1. Process Exits (Free up capital)
        still_open = []
        for pos in active_positions:
            if pos["Exit_Date"] <= current_date:
                profit = pos["Invested"] * pos["Return"]
                cash += (pos["Invested"] + profit)
            else:
                still_open.append(pos)
        active_positions = still_open
        
        # 2. Process New Entries
        while trade_idx < num_potential and trades_df.iloc[trade_idx]["Entry_Date"] == current_date:
            trade = trades_df.iloc[trade_idx]
            trade_idx += 1
            
            holding_tickers = [p["Ticker"] for p in active_positions]
            if trade["Ticker"] in holding_tickers:
                continue # Do not buy a stock we already hold
                
            if len(active_positions) < max_positions:
                invested_amount = cash * allocation_per_trade
                cash -= invested_amount
                
                active_positions.append({
                    "Ticker": trade["Ticker"],
                    "Exit_Date": trade["Exit_Date"],
                    "Invested": invested_amount,
                    "Return": trade["Return"]
                })
                executed_trades.append(trade)
                
        # Record daily equity (Cash + Initial Capital locked in open trades)
        current_equity = cash + sum(p["Invested"] for p in active_positions)
        equity_history.append({"Date": current_date, "Equity": current_equity})

    # ---------------------------------------------------------
    # STEP 3: Generate Realistic Statistics
    # ---------------------------------------------------------
    exec_df = pd.DataFrame(executed_trades)
    
    if exec_df.empty:
        print("No trades were executed under the capital constraints.")
        return
        
    exec_df.to_csv("backtest_trades.csv", index=False)
    
    total_trades = len(exec_df)
    winning_trades = exec_df[exec_df["Return"] > 0]
    losing_trades = exec_df[exec_df["Return"] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    avg_win = winning_trades["Return"].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades["Return"].mean() if not losing_trades.empty else 0
    
    eq_df = pd.DataFrame(equity_history).set_index("Date")
    final_equity = eq_df["Equity"].iloc[-1]
    
    days_in_market = (eq_df.index.max() - eq_df.index.min()).days or 1
    cagr = (final_equity / starting_capital) ** (365.25 / days_in_market) - 1
    max_dd = calculate_drawdown(eq_df["Equity"])
    
    daily_returns = eq_df["Equity"].pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252)

    # Output Report
    print("\n" + "="*50)
    print("📊 REALISTIC OOS BACKTEST (Max 20 Positions, 5% Size)")
    print("="*50)
    print(f"Total Trades Executed: {total_trades} (Down from {num_potential} signals)")
    print(f"Win Rate:              {win_rate:.2%}")
    print(f"Average Winner:        +{avg_win:.2%}")
    print(f"Average Loser:         {avg_loss:.2%}")
    print(f"Average Bars Held:     {exec_df['Bars_Held'].mean():.1f} days")
    print("-" * 50)
    print(f"Starting Capital:      ₹{starting_capital:,.2f}")
    print(f"Ending Capital:        ₹{final_equity:,.2f}")
    print(f"CAGR:                  {cagr:.2%}")
    print(f"Max Drawdown:          {max_dd:.2%}")
    print(f"Sharpe Ratio:          {sharpe:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_backtest()
