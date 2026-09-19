import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized momentum and breakout feature extraction."""
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    grouped = df.groupby('Ticker')

    # 1. Moving Averages & Trend Alignment
    df['EMA_20'] = grouped['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA_50'] = grouped['Close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['Trend_Score'] = (df['Close'] / df['EMA_20'] - 1) + (df['EMA_20'] / df['EMA_50'] - 1)

    # 2. Breakout Proximity (Proximity to 52-Week / 252-day High)
    df['High_252'] = grouped['High'].transform(lambda x: x.rolling(252, min_periods=50).max())
    df['Dist_to_High'] = df['Close'] / df['High_252']

    # 3. Volume Expansion (Volume Surge Indicator)
    df['Vol_SMA_20'] = grouped['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA_20'] + 1e-6)

    # 4. Volatility & Momentum (ATR Proxy & 20-day ROC)
    df['ROC_20'] = grouped['Close'].transform(lambda x: x.pct_change(20))
    df['Range_Ratio'] = (df['High'] - df['Low']) / df['Close']

    # 5. Target Definition: +20% gain in next 20 bars without hitting -7% stop
    future_max_20 = grouped['High'].transform(lambda x: x.shift(-20).rolling(20, min_periods=1).max())
    future_min_20 = grouped['Low'].transform(lambda x: x.shift(-20).rolling(20, min_periods=1).min())

    gain_20 = (future_max_20 - df['Close']) / df['Close'] >= 0.20
    loss_7 = (future_min_20 - df['Close']) / df['Close'] <= -0.07

    # Target = 1 if hit 20%+ target before/without severe breakdown
    df['Target'] = (gain_20 & ~loss_7).astype(int)

    return df.dropna()

def main():
    print("Loading data...")
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    
    print("Extracting features...")
    data = engineer_features(df)
    
    features = [
        'Trend_Score', 'Dist_to_High', 'Vol_Ratio', 
        'ROC_20', 'Range_Ratio'
    ]
    
    # Time-based train/test split to prevent forward-looking bias
    data['Date'] = pd.to_datetime(data['Date'])
    split_date = data['Date'].max() - pd.Timedelta(days=180)
    
    train = data[data['Date'] < split_date]
    test = data[data['Date'] >= split_date]

    X_train, y_train = train[features], train['Target']
    X_test, y_test = test[features], test['Target']

    # Handle class imbalance: 20%+ breakouts occur in < 5% of trading bars
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)

    print(f"Training LightGBM on {len(X_train)} samples...")
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    preds_prob = model.predict_proba(X_test)[:, 1]
    # Filter top decile predictions for high-conviction entries
    threshold = np.quantile(preds_prob, 0.99)
    preds = (preds_prob >= threshold).astype(int)

    print(f"\n--- Out-of-Sample Performance (Threshold: {threshold:.3f}) ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, preds_prob):.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    # Save model artifact
    joblib.dump(model, "breakout_lgbm_model.pkl")
    print("Model saved to breakout_lgbm_model.pkl")

if __name__ == "__main__":
    main()
