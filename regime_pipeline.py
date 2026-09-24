import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------
# 1. DATA INGESTION & FEATURE ENGINEERING
# ---------------------------------------------------------
print("Downloading Nifty 50 data...")
df = yf.download("^NSEI", period="max", auto_adjust=True)

# Flatten multi-index columns if returned by yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna().copy()

# Feature computation
df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
df["realized_vol_21"] = df["log_ret"].rolling(window=21).std() * np.sqrt(252)

# Garman-Klass Volatility (incorporates Open, High, Low, Close)
log_hl = (np.log(df["High"] / df["Low"])) ** 2
log_co = (np.log(df["Close"] / df["Open"])) ** 2
df["gk_vol"] = np.sqrt((0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(21).mean() * 252)

# Momentum & Trend
df["sma_50"] = df["Close"].rolling(50).mean()
df["sma_200"] = df["Close"].rolling(200).mean()
df["dist_sma_200"] = (df["Close"] - df["sma_200"]) / df["sma_200"]
df["rsi_14"] = 100 - (100 / (1 + df["log_ret"].apply(lambda x: x if x > 0 else 0).rolling(14).mean() /
                            (df["log_ret"].apply(lambda x: abs(x) if x < 0 else 0).rolling(14).mean() + 1e-9)))

feature_cols = ["log_ret", "realized_vol_21", "gk_vol", "dist_sma_200", "rsi_14"]
df = df.dropna(subset=feature_cols).copy()

# ---------------------------------------------------------
# 2. SEQUENCE CREATION (SLIDING WINDOW)
# ---------------------------------------------------------
SEQ_LEN = 20
scaler = RobustScaler()
scaled_features = scaler.fit_transform(df[feature_cols].values)

X_sequences = []
for i in range(len(scaled_features) - SEQ_LEN):
    X_sequences.append(scaled_features[i : i + SEQ_LEN])

X_tensor = torch.tensor(np.array(X_sequences), dtype=torch.float32)

# ---------------------------------------------------------
# 3. DEEP LEARNING MODEL: TEMPORAL LSTM AUTOENCODER
# ---------------------------------------------------------
# Captures non-linear dynamic latent market representations
class TemporalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)
        
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        latent = self.fc_enc(h_n[-1])
        
        rep = self.fc_dec(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(rep)
        reconstruction = self.output_layer(dec_out)
        return reconstruction, latent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TemporalAutoencoder(input_dim=len(feature_cols), hidden_dim=32, latent_dim=8).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

print("Training Temporal Autoencoder...")
model.train()
for epoch in range(15):
    for batch in loader:
        x_batch = batch[0].to(device)
        optimizer.zero_grad()
        recon, _ = model(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()

# Extract Latent Representations
model.eval()
with torch.no_grad():
    _, latent_vectors = model(X_tensor.to(device))
    latent_np = latent_vectors.cpu().numpy()

# ---------------------------------------------------------
# 4. REGIME CLUSTERING & ORDERING
# ---------------------------------------------------------
# Gaussian Mixture Model on the deep latent space (3 Regimes: Bear, Sideways, Bull)
N_REGIMES = 3
gmm = GaussianMixture(n_components=N_REGIMES, covariance_type="full", random_state=42)
regimes = gmm.fit_predict(latent_np)

eval_df = df.iloc[SEQ_LEN:].copy()
eval_df["regime"] = regimes

# Map clusters by average forward returns to ensure: 0=Bear, 1=Sideways, 2=Bull
regime_returns = eval_df.groupby("regime")["log_ret"].mean().sort_values()
regime_order = {old_id: new_id for new_id, old_id in enumerate(regime_returns.index)}
eval_df["mapped_regime"] = eval_df["regime"].map(regime_order)

# ---------------------------------------------------------
# 5. BACKTESTING & CAGR COMPUTATION
# ---------------------------------------------------------
# Strategy: Long Nifty in Bull (2) and Sideways (1), Cash in Bear (0)
# Shift regime signal by 1 day to prevent lookahead bias
eval_df["signal"] = (eval_df["mapped_regime"].shift(1) > 0).astype(float)
eval_df["strategy_ret"] = eval_df["signal"] * eval_df["log_ret"]

eval_df["cum_nifty"] = np.exp(eval_df["log_ret"].cumsum())
eval_df["cum_strategy"] = np.exp(eval_df["strategy_ret"].cumsum())

n_years = len(eval_df) / 252.0
nifty_cagr = (eval_df["cum_nifty"].iloc[-1]) ** (1.0 / n_years) - 1.0
strat_cagr = (eval_df["cum_strategy"].iloc[-1]) ** (1.0 / n_years) - 1.0

print(f"Backtest Duration: {n_years:.2f} years")
print(f"Nifty 50 Buy & Hold CAGR: {nifty_cagr * 100:.2f}%")
print(f"Regime-Switching Strategy CAGR: {strat_cagr * 100:.2f}%")

# ---------------------------------------------------------
# 6. VISUALIZATION
# ---------------------------------------------------------
plt.figure(figsize=(14, 7))
plt.plot(eval_df.index, eval_df["cum_nifty"], label=f"Nifty 50 Benchmark (CAGR: {nifty_cagr*100:.2f}%)", color="grey", alpha=0.7)
plt.plot(eval_df.index, eval_df["cum_strategy"], label=f"Deep Regime Strategy (CAGR: {strat_cagr*100:.2f}%)", color="blue", linewidth=1.8)

plt.yscale("log")
plt.title("Nifty 50 vs. Deep Learning Market Regime Strategy (Log Scale)", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Returns (Base = 1.0)", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(loc="upper left", fontsize=11)
plt.tight_layout()

plt.savefig("nifty_regime_cagr.png", dpi=300)
print("Plot successfully saved to nifty_regime_cagr.png")
