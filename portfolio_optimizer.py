import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import PPO
from train_tier1_lgbm import engineer_features
import itertools
import warnings

warnings.filterwarnings("ignore")

def calculate_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1.0
    return drawdown.min()

def run_optimizer():
    print("Loading data and pre-calculating all potential trade outcomes (this takes a few seconds)...")
    raw_df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(raw_df)
    
    lgbm = joblib.load("tier1_lgbm_model.pkl")
    thresh = joblib.load("tier1_threshold.pkl")
    ppo = PPO.load("tier2_ppo_agent.zip")
    
    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    
    # Restrict to last 2 years Out-of-Sample
    data["Date"] = pd.to_datetime(data["Date"])
    start_date = data["Date"].max() - pd.Timedelta(days=730)
    bt_data = data[data["Date"] >= start_date].copy()
    
    bt_data["Prob"] = lgbm.predict_proba(bt_data[feature_cols])[:, 1]
    bt_data["Signal"] = (bt_data["Prob"] >= thresh).astype(int)
    
    signals = bt_data[bt_data["Signal"] == 1].sort_values("Date")
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
            
            if unrealized_pnl <= -0.06:  # Hard stop
                break
                
            obs = np.array([
                unrealized_pnl, min(1.0, step_idx / 15.0),
                row["ATR_14"], row["ROC_20"], row["Dist_to_High"]
            ], dtype=np.float32)
            
            action, _ = ppo.predict(obs, deterministic=True)
            if action == 1: 
                break
                
        realized_return = ((row["Close"] - entry_price) / entry_price) - fee
        potential_trades.append({
            "Entry_Date": entry_date, "Exit_Date": row["Date"],
            "Ticker": ticker, "Return": realized_return, "Bars_Held": step_idx
        })

    # Merge features back in for ranking comparisons
    base_trades_df = pd.DataFrame(potential_trades)
    base_trades_df = pd.merge(base_trades_df, bt_data[['Date', 'Ticker', 'ROC_20', 'Vol_Ratio']], 
                              left_on=['Entry_Date', 'Ticker'], right_on=['Date', 'Ticker'], how='left')
    
    all_dates = sorted(bt_data["Date"].unique())
    
    # Grid Search Parameters to Test
    position_limits = [10, 15, 20, 25]
    ranking_methods = ['Chronological', 'ROC_20', 'Vol_Ratio']
    
    results = []
    
    print("\nSimulating portfolio combinations...")
    for max_pos, rank_method in itertools.product(position_limits, ranking_methods):
        starting_capital = 1_000_000.0
        cash = starting_capital
        allocation = 1.0 / max_pos  # Dynamically set allocation size based on max positions
        
        active_positions = []
        executed_trades = []
        equity_history = []
        
        for current_date in all_dates:
            # 1. Process Exits
            still_open = []
            for pos in active_positions:
                if pos["Exit_Date"] <= current_date:
                    profit = pos["Invested"] * pos["Return"]
                    cash += (pos["Invested"] + profit)
                else:
                    still_open.append(pos)
            active_positions = still_open
            
            # 2. Get today's signals
            todays_signals = base_trades_df[base_trades_df["Entry_Date"] == current_date]
            if not todays_signals.empty:
                # Rank signals if cash is limited
                if rank_method == 'ROC_20':
                    todays_signals = todays_signals.sort_values(by="ROC_20", ascending=False)
                elif rank_method == 'Vol_Ratio':
                    todays_signals = todays_signals.sort_values(by="Vol_Ratio", ascending=False)
                
                for _, trade in todays_signals.iterrows():
                    if trade["Ticker"] in [p["Ticker"] for p in active_positions]:
                        continue 
                        
                    if len(active_positions) < max_pos:
                        invested_amount = cash * allocation
                        cash -= invested_amount
                        active_positions.append({
                            "Ticker": trade["Ticker"], "Exit_Date": trade["Exit_Date"],
                            "Invested": invested_amount, "Return": trade["Return"]
                        })
                        executed_trades.append(trade)
                    else:
                        break # Out of capital limit
                    
            current_equity = cash + sum(p["Invested"] for p in active_positions)
            equity_history.append({"Date": current_date, "Equity": current_equity})

        # Calculate Combination Metrics
        exec_df = pd.DataFrame(executed_trades)
        if exec_df.empty:
            continue
            
        win_rate = len(exec_df[exec_df["Return"] > 0]) / len(exec_df)
        eq_df = pd.DataFrame(equity_history).set_index("Date")
        days_in_market = (eq_df.index.max() - eq_df.index.min()).days or 1
        cagr = (eq_df["Equity"].iloc[-1] / starting_capital) ** (365.25 / days_in_market) - 1
        max_dd = calculate_drawdown(eq_df["Equity"])
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        results.append({
            "Max_Pos": max_pos,
            "Ranking": rank_method,
            "Trades": len(exec_df),
            "Win_Rate%": round(win_rate * 100, 2),
            "CAGR%": round(cagr * 100, 2),
            "Max_DD%": round(max_dd * 100, 2),
            "Calmar": round(calmar, 2)
        })
        
    # Display the final leaderboard
    results_df = pd.DataFrame(results).sort_values(by="Calmar", ascending=False)
    print("\n" + "="*80)
    print("🏆 PORTFOLIO OPTIMIZATION LEADERBOARD (Sorted by Calmar Ratio)")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv("portfolio_optimization_results.csv", index=False)

if __name__ == "__main__":
    run_optimizer()
