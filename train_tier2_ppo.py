import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import joblib
import warnings

warnings.filterwarnings("ignore")

class TrendFollowingEnv(gym.Env):
    def __init__(self, data_slices):
        super(TrendFollowingEnv, self).__init__()
        self.data_slices = data_slices
        self.current_slice = None
        self.current_step = 0
        
        self.action_space = spaces.Discrete(2) # 0 = Hold, 1 = Sell
        
        # [Unrealized_PnL, Normalized_Days_Held, ATR_14, ROC_20, Dist_to_High]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_slice = self.data_slices[np.random.randint(len(self.data_slices))]
        self.current_step = 0
        self.entry_price = self.current_slice.iloc[0]["Close"]
        self.peak_price = self.entry_price
        return self._get_obs(), {}
        
    def _get_obs(self):
        row = self.current_slice.iloc[self.current_step]
        unrealized_pnl = (row["Close"] - self.entry_price) / self.entry_price
        
        # Normalizing holding period to a massive 120-day window instead of 15
        days_held = min(1.0, self.current_step / 120.0)
        
        return np.array([
            unrealized_pnl, days_held,
            row["ATR_14"], row["ROC_20"], row["Dist_to_High"]
        ], dtype=np.float32)
        
    def step(self, action):
        row = self.current_slice.iloc[self.current_step]
        close = row["Close"]
        self.peak_price = max(self.peak_price, close)
        unrealized_pnl = (close - self.entry_price) / self.entry_price
        
        reward = 0
        terminated = False
        
        # Hard stop protection (Capital Preservation)
        if unrealized_pnl <= -0.08:
            reward = -5.0 # Massive penalty for holding a losing trade too long
            terminated = True
            
        elif action == 1: # Agent decides to SELL
            terminated = True
            # The Probability Paradox fix: Penalize selling if momentum is still surging
            if row["ROC_20"] > 3.0:
                reward = -10.0 # Severe punishment for cutting a massive winner early
            else:
                # Reward for locking in profit when momentum dies
                reward = unrealized_pnl * 20.0
                
        else: # Agent decides to HOLD
            # Reward for sitting on hands while momentum is positive
            if row["ROC_20"] > 0:
                reward = 1.0 
            # Penalize for holding during a severe drawdown from the peak
            drawdown_from_peak = (self.peak_price - close) / self.peak_price
            if drawdown_from_peak > 0.15:
                reward = -2.0
                
        self.current_step += 1
        
        # End of Episode (Trade completes 120 days)
        if self.current_step >= len(self.current_slice) - 1:
            terminated = True
            if action == 0:
                reward = unrealized_pnl * 20.0
                
        return self._get_obs(), reward, terminated, False, {}

def train_ppo():
    print("Loading Nifty 500 data to generate training environments...")
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    
    print("Generating historical breakouts for the RL Agent to practice on...")
    lgbm = joblib.load("tier1_lgbm_model.pkl")
    thresh = joblib.load("tier1_threshold.pkl")
    
    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    df["Prob"] = lgbm.predict_proba(df[feature_cols])[:, 1]
    df["Signal"] = (df["Prob"] >= thresh).astype(int)
    
    signals = df[df["Signal"] == 1]
    ticker_groups = dict(tuple(df.groupby("Ticker")))
    
    data_slices = []
    for _, sig in signals.iterrows():
        t_data = ticker_groups[sig["Ticker"]]
        # Give the agent a 120-day horizon to learn massive trends
        trade_slice = t_data[t_data["Date"] >= sig["Date"]].head(120).reset_index(drop=True)
        if len(trade_slice) >= 15:
            data_slices.append(trade_slice)
            
    print(f"Created {len(data_slices)} historical breakout environments.")
    
    env = TrendFollowingEnv(data_slices)
    
    print("Training PPO Agent to hunt multibaggers (150,000 steps)...")
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005, n_steps=2048)
    model.learn(total_timesteps=150000)
    
    model.save("tier2_ppo_agent.zip")
    print("Saved upgraded neural network to tier2_ppo_agent.zip")

if __name__ == "__main__":
    train_ppo()
