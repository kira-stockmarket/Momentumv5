import os
import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from train_tier1_lgbm import engineer_features

def main():
    print("Starting daily live execution run...")
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(df)

    lgbm = joblib.load("tier1_lgbm_model.pkl")
    thresh = joblib.load("tier1_threshold.pkl")
    ppo = PPO.load("tier2_ppo_agent.zip")

    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    
    # 1. Screen Today's Top Breakout Entries
    latest = data.sort_values("Date").groupby("Ticker").last().reset_index()
    latest["Prob"] = lgbm.predict_proba(latest[feature_cols])[:, 1]
    
    candidates = latest[latest["Prob"] >= thresh].sort_values(by="Prob", ascending=False)
    
    # Prepare Markdown Report
    lines = []
    lines.append("# 📈 Daily Quantitative Trading Action Report")
    lines.append(f"**Market Breadth**: {latest['Market_Breadth'].iloc[0]:.1%} stocks above 20-EMA\n")
    lines.append("## 🟢 Today's Fresh Entry Candidates (Tier 1 Breakouts)")

    if candidates.empty:
        lines.append("_No tickers cleared the high-conviction breakout threshold today._\n")
    else:
        top_candidates = candidates.head(5)[["Ticker", "Close", "Prob", "ATR_14"]]
        lines.append("| Ticker | Close | Win Probability | ATR % | Recommended Action |")
        lines.append("|---|---|---|---|---|")
        for _, row in top_candidates.iterrows():
            lines.append(f"| **{row['Ticker']}** | ₹{row['Close']:.2f} | {row['Prob']:.1%} | {row['ATR_14']*100:.2f}% | BUY (Target +10%, Cut via RL) |")
        lines.append("")

    # 2. Evaluate Active Trades with RL Agent (if an active portfolio file exists)
    portfolio_file = "active_positions.csv"
    lines.append("## 🛡️ Active Position Management (Tier 2 RL Exit Signals)")
    
    if os.path.exists(portfolio_file):
        positions = pd.read_csv(portfolio_file)
        lines.append("| Ticker | Entry Price | Current Price | Unrealized PnL | RL Recommendation |")
        lines.append("|---|---|---|---|---|")
        
        for idx, pos in positions.iterrows():
            ticker = pos["Ticker"]
            entry_price = float(pos["Entry_Price"])
            bars_held = int(pos.get("Bars_Held", 1))
            
            ticker_data = latest[latest["Ticker"] == ticker]
            if not ticker_data.empty:
                current_price = float(ticker_data["Close"].iloc[0])
                unrealized_pnl = (current_price - entry_price) / entry_price
                
                obs = np.array([
                    unrealized_pnl,
                    min(1.0, bars_held / 15.0),
                    ticker_data["ATR_14"].iloc[0],
                    ticker_data["ROC_20"].iloc[0],
                    ticker_data["Dist_to_High"].iloc[0]
                ], dtype=np.float32)
                
                action, _ = ppo.predict(obs, deterministic=True)
                rec = "🔴 **CLOSE POSITION NOW**" if action == 1 else "🟢 **HOLD (Trail Stop)**"
                lines.append(f"| {ticker} | ₹{entry_price:.2f} | ₹{current_price:.2f} | {unrealized_pnl:.2%} | {rec} |")
        lines.append("")
    else:
        lines.append("_No active portfolio tracking file (`active_positions.csv`) found._\n")

    report_content = "\n".join(lines)
    with open("daily_signals.md", "w") as f:
        f.write(report_content)
    print(report_content)

if __name__ == "__main__":
    main()
