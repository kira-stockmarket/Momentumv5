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
    
    # Restrict backtest to the last 2 years (approx 500 trading days) to represent Out-of-Sample
    data["Date"] = pd.to_datetime(data["Date"])
    start_date = data["Date"].max() - pd.Timedelta(days=730)
    bt_data = data[data["Date"] >= start_date].copy()
    
    print(f"Generating Tier 1 Signals from {start_date.date()} to {bt_data['Date'].max().date()}...")
    bt_data["Prob"] = lgbm.predict_proba(bt_data[feature_cols])[:, 1]
    bt_data["Signal"] = (bt_data["Prob"] >= thresh).astype(int)
    
    signals = bt_data[bt_data["Signal"] == 1].sort_values("Date")
    
    completed_trades = []
    active_holds = {}  # Tracks tickers currently held to avoid overlapping entries
    
    print(f"Simulating Tier 2 RL Executions on {len(signals)} raw signals...")
    
    # Group data by ticker for fast slice lookups
    ticker_groups = dict(tuple(bt_data.groupby("Ticker")))
    
    for _, sig in signals.iterrows():
        ticker = sig["Ticker"]
        entry_date = sig["Date"]
        
        # Skip if we are already holding this stock
        if ticker in active_holds and entry_date <= active_holds[ticker]:
            continue
            
        t_data = ticker_groups[ticker]
        # Get the next 15 days of price action for this ticker
        trade_slice = t_data[t_data["Date"] >= entry_date].head(16).reset_index(drop=True)
        
        if len(trade_slice) < 2:
            continue
            
        entry_price = trade_slice.iloc[0]["Close"]
        exit_price = entry_price
        exit_date = entry_date
        bars_held = 0
        fee = 0.001 # 0.1% slippage + friction per trade
        
        # Step through the trade slice with the RL agent
        for step_idx in range(1, len(trade_slice)):
            row = trade_slice.iloc[step_idx]
            unrealized_pnl = (row["Close"] - entry_price) / entry_price
            bars_held = step_idx
            exit_date = row["Date"]
            exit_price = row["Close"]
            
            # Hard stop safety guard (RL agent was trained with this)
            if unrealized_pnl <= -0.06:
                break
                
            # RL Agent Observation
            obs = np.array([
                unrealized_pnl,
                min(1.0, step_idx / 15.0),
                row["ATR_14"],
                row["ROC_20"],
                row["Dist_to_High"]
            ], dtype=np.float32)
            
            action, _ = ppo.predict(obs, deterministic=True)
            
            if action == 1: # Close Position
                break
                
        # Record trade result
        realized_return = ((exit_price - entry_price) / entry_price) - fee
        completed_trades.append({
            "Entry_Date": entry_date,
            "Exit_Date": exit_date,
            "Ticker": ticker,
            "Return": realized_return,
            "Bars_Held": bars_held
        })
        active_holds[ticker] = exit_date

    trades_df = pd.DataFrame(completed_trades)
    
    if trades_df.empty:
        print("No trades were executed in the backtest period.")
        return
        
    trades_df = trades_df.sort_values("Exit_Date")
    
    # ---- Calculate Statistics ----
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df["Return"] > 0]
    losing_trades = trades_df[trades_df["Return"] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    avg_win = winning_trades["Return"].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades["Return"].mean() if not losing_trades.empty else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Portfolio Simulation (Assume starting equity ₹1,000,000 and allocating 5% per trade)
    capital = 1_000_000
    equity_curve = []
    
    for _, trade in trades_df.iterrows():
        position_size = capital * 0.05
        trade_profit = position_size * trade["Return"]
        capital += trade_profit
        equity_curve.append({"Date": trade["Exit_Date"], "Equity": capital})
        
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    # Resample to daily to calculate annualized metrics smoothly
    eq_df = eq_df.resample("D").last().ffill().dropna()
    
    final_equity = capital
    days_in_market = (eq_df.index.max() - eq_df.index.min()).days or 1
    cagr = (final_equity / 1_000_000) ** (365.25 / days_in_market) - 1
    max_dd = calculate_drawdown(eq_df["Equity"])
    
    daily_returns = eq_df["Equity"].pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252)

    # ---- Output Report ----
    print("\n" + "="*50)
    print("📊 OUT-OF-SAMPLE BACKTEST RESULTS (Last 2 Years)")
    print("="*50)
    print(f"Total Trades Executed: {total_trades}")
    print(f"Win Rate:              {win_rate:.2%}")
    print(f"Average Winner:        +{avg_win:.2%}")
    print(f"Average Loser:         {avg_loss:.2%}")
    print(f"Average Bars Held:     {trades_df['Bars_Held'].mean():.1f} days")
    print(f"Mathematical Expectancy:{expectancy:.4f} per trade")
    print("-" * 50)
    print(f"Starting Capital:      ₹1,000,000")
    print(f"Ending Capital:        ₹{final_equity:,.2f}")
    print(f"CAGR:                  {cagr:.2%}")
    print(f"Max Drawdown:          {max_dd:.2%}")
    print(f"Sharpe Ratio:          {sharpe:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_backtest()
