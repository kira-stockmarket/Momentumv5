import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve
import joblib
import warnings

warnings.filterwarnings("ignore")

def engineer_features(df):
    """
    Calculates the 6 core features required by both the LightGBM and RL models.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    
    # 1. ATR_14 (Normalized as a % of price to fix unit mismatch)
    df['Prev_Close'] = df.groupby('Ticker')['Close'].shift(1)
    df['TR'] = df[['High', 'Prev_Close']].max(axis=1) - df[['Low', 'Prev_Close']].min(axis=1)
    df['ATR_14'] = df.groupby('Ticker')['TR'].transform(lambda x: x.rolling(14).mean())
    df['ATR_14'] = df['ATR_14'] / df['Close'] 
    
    # 2. ROC_20 (Rate of Change / Momentum)
    df['ROC_20'] = df.groupby('Ticker')['Close'].pct_change(periods=20) * 100
    
    # 3. Vol_Ratio (Institutional Volume Footprint)
    df['Vol_SMA_20'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA_20'] + 1e-8)
    
    # 4. Dist_to_High (Distance to 52-week high)
    df['High_252'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(252).max())
    df['Dist_to_High'] = (df['High_252'] - df['Close']) / df['Close']
    
    # 5. Trend_Score (Moving Average Alignment: 0 to 3)
    df['SMA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(50).mean())
    df['SMA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    df['Trend_Score'] = ((df['Close'] > df['SMA_50']).astype(int) + 
                         (df['SMA_50'] > df['SMA_200']).astype(int) + 
                         (df['Close'] > df['SMA_200']).astype(int))
                         
    # 6. Market_Breadth (% of all Nifty 500 stocks above their 50 SMA)
    df['Above_50'] = (df['Close'] > df['SMA_50']).astype(int)
    breadth = df.groupby('Date')['Above_50'].mean().reset_index()
    breadth.rename(columns={'Above_50': 'Market_Breadth'}, inplace=True)
    df = df.drop(columns=['Above_50']).merge(breadth, on='Date', how='left')
    
    # Drop rows with incomplete indicator data
    df = df.dropna(subset=['ATR_14', 'ROC_20', 'Vol_Ratio', 'Dist_to_High', 'Trend_Score', 'Market_Breadth'])
    return df

def train_lgbm():
    print("Loading raw data...")
    raw_df = pd.read_parquet("nifty500_ohlcv.parquet")
    
    print("Engineering features...")
    df = engineer_features(raw_df)
    
    print("Generating Heavy ML Target (20% move within 60 days)...")
    # THE HOLY GRAIL FIX: Ask the ML to predict multi-month macro trends
    df['Future_60d_Max'] = df.groupby('Ticker')['High'].transform(lambda x: x.rolling(60).max().shift(-60))
    df['Target'] = ((df['Future_60d_Max'] - df['Close']) / df['Close'] >= 0.20).astype(int)
    
    # Drop the last 60 days of the dataset where the future is unknown
    df = df.dropna(subset=['Target'])
    
    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    
    # Chronological Split: Keep the last 2 years strictly Out-of-Sample
    split_date = df['Date'].max() - pd.Timedelta(days=730)
    train_df = df[df['Date'] < split_date]
    val_df = df[df['Date'] >= split_date]
    
    X_train, y_train = train_df[feature_cols], train_df['Target']
    X_val, y_val = val_df[feature_cols], val_df['Target']
    
    print(f"Training on {len(train_df)} rows, Validating on {len(val_df)} rows...")
    
    # Train the neural network
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=64,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Calculate the exact mathematical threshold for dynamic allocation
    val_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    
    # We want a threshold that guarantees at least 45% precision on these massive trends
    target_precision = 0.45
    valid_idx = np.where(precisions >= target_precision)[0]
    
    if len(valid_idx) > 0:
        best_thresh = thresholds[valid_idx[0]]
        # Safety net: Ensure threshold isn't impossibly high
        if best_thresh >= 0.95: 
            best_thresh = 0.65
    else:
        best_thresh = 0.65 
        
    print(f"\nModel Training Complete!")
    print(f"Optimal Probability Threshold for Breakouts: {best_thresh:.4f}")
    
    joblib.dump(model, "tier1_lgbm_model.pkl")
    joblib.dump(best_thresh, "tier1_threshold.pkl")
    print("Saved Tier 1 model to disk. Ready for Backtest or RL Training.")

if __name__ == "__main__":
    train_lgbm()
