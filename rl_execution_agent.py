import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import joblib
import warnings

warnings.filterwarnings("ignore")

class TradeExecutionEnv(gym.Env):
    """
    Tier-2 Trade Exit Environment.
    Episode begins when a stock is bought on a Tier-1 breakout signal.
    Actions:
      0: HOLD
      1: CLOSE POSITION (Exit now)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, trade_episodes: list, max_holding_bars: int = 15, fee: float = 0.001):
        super().__init__()
        self.trade_episodes = trade_episodes
        self.max_holding_bars = max_holding_bars
        self.fee = fee

        # Action: 0 = Hold, 1 = Close
        self.action_space = spaces.Discrete(2)

        # Observation: [unrealized_pnl, bars_held_normalized, atr_14, roc_20, dist_to_high]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        self.current_episode_idx = 0
        self.step_idx = 0
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_episode_idx = np.random.randint(0, len(self.trade_episodes))
        self.episode_data = self.trade_episodes[self.current_episode_idx]
        self.entry_price = self.episode_data.iloc[0]["Close"]
        self.step_idx = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.episode_data.iloc[self.step_idx]
        unrealized_pnl = (row["Close"] - self.entry_price) / self.entry_price
        bars_held_norm = self.step_idx / float(self.max_holding_bars)
        obs = np.array([
            unrealized_pnl,
            bars_held_norm,
            row.get("ATR_14", 0.02),
            row.get("ROC_20", 0.0),
            row.get("Dist_to_High", 0.95)
        ], dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def step(self, action):
        self.step_idx += 1
        terminated = False
        reward = 0.0

        current_row = self.episode_data.iloc[self.step_idx]
        unrealized_pnl = (current_row["Close"] - self.entry_price) / self.entry_price

        # Automatic hard-stop breach safety guard
        if unrealized_pnl <= -0.06:
            action = 1

        if action == 1:
            # Trade exit
            terminated = True
            realized_pnl = unrealized_pnl - self.fee
            # Asymmetric reward: bonus for locked-in +10% gain, sharp penalty for losses
            if realized_pnl > 0.08:
                reward = realized_pnl * 3.0
            elif realized_pnl < 0:
                reward = realized_pnl * 2.5
            else:
                reward = realized_pnl
        else:
            # Holding reward: small time decay penalty to discourage idle capital
            reward = -0.0005
            if self.step_idx >= min(self.max_holding_bars, len(self.episode_data) - 1):
                terminated = True
                realized_pnl = unrealized_pnl - self.fee
                reward = realized_pnl

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, False, {"pnl": unrealized_pnl}

def generate_episodes(df: pd.DataFrame, max_bars: int = 15):
    """Splits historical breakout signals into trade slice episodes."""
    episodes = []
    signals = df[df["Signal"] == 1]
    for _, sig in signals.iterrows():
        t_data = df[(df["Ticker"] == sig["Ticker"]) & (df["Date"] >= sig["Date"])].head(max_bars + 1)
        if len(t_data) >= 3:
            episodes.append(t_data.reset_index(drop=True))
    return episodes

def train_and_evaluate_rl():
    print("Loading data for Tier 2 RL Execution Agent...")
    from train_tier1_lgbm import engineer_features
    
    raw_df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(raw_df)
    
    lgbm = joblib.load("tier1_lgbm_model.pkl")
    thresh = joblib.load("tier1_threshold.pkl")
    
    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    probs = lgbm.predict_proba(data[feature_cols])[:, 1]
    data["Signal"] = (probs >= thresh).astype(int)

    # Walk-forward split
    data["Date"] = pd.to_datetime(data["Date"])
    split_date = data["Date"].max() - pd.Timedelta(days=180)
    train_data = data[data["Date"] < split_date]
    test_data = data[data["Date"] >= split_date]

    train_episodes = generate_episodes(train_data)
    test_episodes = generate_episodes(test_data)

    print(f"Generated {len(train_episodes)} training episodes and {len(test_episodes)} out-of-sample test episodes.")
    if len(train_episodes) == 0 or len(test_episodes) == 0:
        print("Not enough breakout instances to train RL. Check threshold or data volume.")
        return

    # Train PPO agent
    env = DummyVecEnv([lambda: TradeExecutionEnv(train_episodes)])
    ppo_agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        gamma=0.98,
        gae_lambda=0.92,
        ent_coef=0.01,
        verbose=0
    )

    print("Training PPO policy (30,000 steps)...")
    ppo_agent.learn(total_timesteps=30000)

    # Strict Out-of-Sample Walk-Forward Backtest
    print("\n--- Running Out-of-Sample RL Backtest ---")
    test_env = TradeExecutionEnv(test_episodes)
    oos_trades = []

    for ep_idx in range(min(len(test_episodes), 300)):
        test_env.trade_episodes = [test_episodes[ep_idx]]
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = ppo_agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            if done:
                oos_trades.append(info["pnl"])

    oos_trades = np.array(oos_trades)
    win_rate = np.mean(oos_trades > 0)
    avg_trade = np.mean(oos_trades)
    sharpe = (avg_trade / (np.std(oos_trades) + 1e-8)) * np.sqrt(252 / 15)

    print(f"OOS Win Rate: {win_rate:.1%}")
    print(f"OOS Average Trade Return: {avg_trade:.2%}")
    print(f"OOS Trade Sharpe Ratio: {sharpe:.2f}")

    ppo_agent.save("tier2_ppo_agent.zip")
    print("RL Agent saved to tier2_ppo_agent.zip")

if __name__ == "__main__":
    train_and_evaluate_rl()
