import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, precision_score
import joblib
import warnings

warnings.filterwarnings('ignore')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # 1. Market Breadth Regime (Internal Calculation to gauge overall market health)
    df['Above_EMA20'] = df['Close'] > df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=20).mean())
    breadth = df.groupby('Date')['Above_EMA20'].mean().reset_index()
    breadth.rename(columns={'Above_EMA20': 'Market_Breadth'}, inplace=True)
    df = df.drop(columns=['Above_EMA20']).merge(breadth, on='Date', how='left')

    grouped = df.groupby('Ticker')

    # 2. Moving Averages & Trend Alignment
    df['EMA_20'] = grouped['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EMA_50'] = grouped['Close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['Trend_Score'] = (df['Close'] / df['EMA_20'] - 1) + (df['EMA_20'] / df['EMA_50'] - 1)

    # 3. Volatility & ATR Proxy (Penalizes wild, un-tradeable stocks)
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['ATR_14'] = grouped['Daily_Range'].transform(lambda x: x.rolling(14).mean())
    
    # 4. Breakout Proximity
    df['High_252'] = grouped['High'].transform(lambda x: x.rolling(252, min_periods=50).max())
    df['Dist_to_High'] = df['Close'] / df['High_252']

    # 5. Volume Surge
    df['Vol_SMA_20'] = grouped['Volume'].transform(lambda x: x.rolling(20).mean())
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_SMA_20'] + 1e-6)

    # 6. Momentum
    df['ROC_20'] = grouped['Close'].transform(lambda x: x.pct_change(20))

    # 7. Adjusted Target: +10% within 15 days without hitting -7% stop
    future_max_15 = grouped['High'].transform(lambda x: x.shift(-15).rolling(15, min_periods=1).max())
    future_min_15 = grouped['Low'].transform(lambda x: x.shift(-15).rolling(15, min_periods=1).min())

    gain_10 = (future_max_15 - df['Close']) / df['Close'] >= 0.10
    loss_7 = (future_min_15 - df['Close']) / df['Close'] <= -0.07

    df['Target'] = (gain_10 & ~loss_7).astype(int)

    return df.dropna()

def main():
    print("Loading data...")
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    
    print("Extracting features (including Market Breadth & ATR)...")
    data = engineer_features(df)
    
    features = [
        'Trend_Score', 'Dist_to_High', 'Vol_Ratio', 
        'ROC_20', 'ATR_14', 'Market_Breadth'
    ]
    
    # Forward-walk split (Train on older data, test on recent 6 months)
    data['Date'] = pd.to_datetime(data['Date'])
    split_date = data['Date'].max() - pd.Timedelta(days=180)
    
    train = data[data['Date'] < split_date]
    test = data[data['Date'] >= split_date]

    X_train, y_train = train[features], train['Target']
    X_test, y_test = test[features], test['Target']

    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)

    print(f"Training Heavily Regularized LightGBM on {len(X_train)} samples...")
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.03,        # Slower learning for better generalization
        max_depth=4,               # Shallow trees to prevent memorizing noise
        colsample_bytree=0.8,      # Feature sampling
        subsample=0.8,             # Row sampling (bagging)
        extra_trees=True,          # Adds randomness to split points (crucial for noisy finance data)
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    preds_prob = model.predict_proba(X_test)[:, 1]
    
    # Dynamically find the threshold that achieves at least 35% precision on training data
    train_probs = model.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0.5, 0.99, 100)
    best_thresh = 0.90
    for t in thresholds:
        if precision_score(y_train, (train_probs >= t).astype(int), zero_division=0) >= 0.35:
            best_thresh = t
            break

    preds = (preds_prob >= best_thresh).astype(int)

    print(f"\n--- Out-of-Sample Performance (Threshold: {best_thresh:.3f}) ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, preds_prob):.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, "breakout_lgbm_model.pkl")
    # Save the calibrated threshold alongside the model
    joblib.dump(best_thresh, "model_threshold.pkl")
    print("Model and calibrated threshold saved.")

if __name__ == "__main__":
    main()
