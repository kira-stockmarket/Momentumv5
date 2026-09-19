import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, precision_score
import joblib
import warnings

warnings.filterwarnings("ignore")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # 1. Market Breadth Regime (% of stocks above 20-EMA across the universe)
    df["EMA20_temp"] = df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20).mean())
    df["Above_EMA20"] = df["Close"] > df["EMA20_temp"]
    breadth = df.groupby("Date")["Above_EMA20"].mean().reset_index()
    breadth.rename(columns={"Above_EMA20": "Market_Breadth"}, inplace=True)
    df = df.drop(columns=["EMA20_temp", "Above_EMA20"]).merge(breadth, on="Date", how="left")

    grouped = df.groupby("Ticker")

    # 2. Momentum & Trend Alignment
    df["EMA_20"] = grouped["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df["EMA_50"] = grouped["Close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["Trend_Score"] = (df["Close"] / df["EMA_20"] - 1.0) + (df["EMA_20"] / df["EMA_50"] - 1.0)

    # 3. Volatility & Normalized Average True Range (NATR)
    df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["ATR_14"] = grouped["Daily_Range"].transform(lambda x: x.rolling(14).mean())

    # 4. Proximity to 52-Week High & Volume Expansion
    df["High_252"] = grouped["High"].transform(lambda x: x.rolling(252, min_periods=50).max())
    df["Dist_to_High"] = df["Close"] / df["High_252"]
    df["Vol_SMA20"] = grouped["Volume"].transform(lambda x: x.rolling(20).mean())
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_SMA20"] + 1e-6)
    df["ROC_20"] = grouped["Close"].transform(lambda x: x.pct_change(20))

    # 5. Label: +10% within 15 bars without hitting -5% stop (Positive EV setup)
    future_max_15 = grouped["High"].transform(lambda x: x.shift(-15).rolling(15, min_periods=1).max())
    future_min_15 = grouped["Low"].transform(lambda x: x.shift(-15).rolling(15, min_periods=1).min())

    gain_10 = (future_max_15 - df["Close"]) / df["Close"] >= 0.10
    loss_5 = (future_min_15 - df["Close"]) / df["Close"] <= -0.05

    df["Target"] = (gain_10 & ~loss_5).astype(int)
    return df.dropna().reset_index(drop=True)

def train_lgbm():
    print("Loading data for Tier 1 LightGBM Screener...")
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(df)

    feature_cols = ["Trend_Score", "Dist_to_High", "Vol_Ratio", "ROC_20", "ATR_14", "Market_Breadth"]
    data["Date"] = pd.to_datetime(data["Date"])

    # Forward-split: last 180 days held out for strictly out-of-sample test
    split_date = data["Date"].max() - pd.Timedelta(days=180)
    train = data[data["Date"] < split_date]
    test = data[data["Date"] >= split_date]

    X_train, y_train = train[feature_cols], train["Target"]
    X_test, y_test = test[feature_cols], test["Target"]

    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)
    print(f"Training LightGBM on {len(X_train)} samples (Out-of-sample set: {len(X_test)})...")

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        colsample_bytree=0.8,
        subsample=0.8,
        extra_trees=True,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    # Calibrate decision threshold for >= 40% precision in-sample
    calibrated_thresh = 0.55
    for t in np.linspace(0.50, 0.95, 46):
        if precision_score(y_train, (train_probs >= t).astype(int), zero_division=0) >= 0.40:
            calibrated_thresh = t
            break

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= calibrated_thresh).astype(int)

    print(f"\n--- Tier 1 LightGBM Test Results (Threshold: {calibrated_thresh:.2f}) ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(classification_report(y_test, test_preds, zero_division=0))

    joblib.dump(model, "tier1_lgbm_model.pkl")
    joblib.dump(calibrated_thresh, "tier1_threshold.pkl")
    print("Tier 1 artifacts saved successfully.")

if __name__ == "__main__":
    train_lgbm()
