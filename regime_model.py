import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. DATA INGESTION & FEATURE ENGINEERING
# ---------------------------------------------------------
print("Downloading Nifty 50 data...")
df = yf.download("^NSEI", period="max", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna().copy()

# HMMs prefer low-dimensional, stationary features. 
# We use log returns and 21-day rolling volatility.
df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["vol_21"] = df["log_ret"].rolling(window=21).std() * np.sqrt(252)

df = df.dropna().copy()
X = df[["log_ret", "vol_21"]].values

# ---------------------------------------------------------
# 2. HMM REGIME DETECTION
# ---------------------------------------------------------
print("Training Hidden Markov Model (2 States)...")
# 2 States: Typically isolates into "Low Vol Bull" and "High Vol Bear"
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
hmm_model.fit(X)

# Predict the hidden states for the entire sequence
df["regime"] = hmm_model.predict(X)

# Dynamically identify which state is the Bull (Risk-On) vs Bear (Risk-Off)
# The state with the lower volatility is almost always the Bull regime in equities.
regime_vol = df.groupby("regime")["vol_21"].mean()
bull_regime_id = regime_vol.idxmin()

# Map: 1 for Bull (Invested), 0 for Bear (Cash)
df["mapped_regime"] = np.where(df["regime"] == bull_regime_id, 1, 0)

# ---------------------------------------------------------
# 3. BACKTESTING WITH CASH YIELD
# ---------------------------------------------------------
# Shift signal by 1 day to prevent look-ahead bias
df["signal"] = df["mapped_regime"].shift(1)
df = df.dropna().copy()

# Indian Risk-Free Rate assumption (e.g., Liquid BeES / TREPS at ~6% annualized)
rf_daily = np.log(1 + 0.06) / 252.0

# When signal is 1, hold Nifty. When signal is 0, hold 6% yielding cash.
df["strategy_ret"] = np.where(df["signal"] == 1.0, df["log_ret"], rf_daily)

df["cum_nifty"] = np.exp(df["log_ret"].cumsum())
df["cum_strategy"] = np.exp(df["strategy_ret"].cumsum())

# ---------------------------------------------------------
# 4. PERFORMANCE METRICS
# ---------------------------------------------------------
n_years = len(df) / 252.0
nifty_cagr = (df["cum_nifty"].iloc[-1]) ** (1.0 / n_years) - 1.0
strat_cagr = (df["cum_strategy"].iloc[-1]) ** (1.0 / n_years) - 1.0

print(f"\n--- PERFORMANCE OVER {n_years:.2f} YEARS ---")
print(f"Nifty 50 Buy & Hold CAGR:   {nifty_cagr * 100:.2f}%")
print(f"HMM Regime Strategy CAGR:   {strat_cagr * 100:.2f}%")

# Calculate Max Drawdown and Sharpe Ratio
for col, name in [("cum_nifty", "Nifty 50"), ("cum_strategy", "HMM Strategy")]:
    cum = df[col]
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() * 100
    
    daily_ret = df["log_ret"] if name == "Nifty 50" else df["strategy_ret"]
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    
    print(f"\n[{name}]")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")

# ---------------------------------------------------------
# 5. VISUALIZATION
# ---------------------------------------------------------
plt.figure(figsize=(14, 7))

# Plot Equity Curves
plt.plot(df.index, df["cum_nifty"], label=f"Nifty 50 (CAGR: {nifty_cagr*100:.2f}%)", color="grey", alpha=0.7)
plt.plot(df.index, df["cum_strategy"], label=f"HMM Strategy (CAGR: {strat_cagr*100:.2f}%)", color="blue", linewidth=1.8)

# Highlight Bear Regimes in Red
bear_dates = df[df["signal"] == 0].index
for date in bear_dates:
    plt.axvline(x=date, color='red', alpha=0.02)

plt.yscale("log")
plt.title("Nifty 50 vs. HMM Market Regime Strategy (Log Scale)\nRed Background = Cash Regime (Earning 6%)", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Returns (Base = 1.0)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(loc="upper left", fontsize=11)
plt.tight_layout()

plt.savefig("nifty_regime_cagr.png", dpi=300)
print("\nPlot successfully saved to nifty_regime_cagr.png")
