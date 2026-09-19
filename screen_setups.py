import pandas as pd
import joblib
import warnings
from train_model import engineer_features

warnings.filterwarnings('ignore')

def screen_today():
    model = joblib.load("breakout_lgbm_model.pkl")
    threshold = joblib.load("model_threshold.pkl")
    
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(df)
    
    features = [
        'Trend_Score', 'Dist_to_High', 'Vol_Ratio', 
        'ROC_20', 'ATR_14', 'Market_Breadth'
    ]
    
    # Isolate the most recent trading day
    latest = data.sort_values('Date').groupby('Ticker').last().reset_index()
    latest['Probability'] = model.predict_proba(latest[features])[:, 1]
    latest['Signal'] = (latest['Probability'] >= threshold).astype(int)
    
    # Filter for actionable signals and sort by highest conviction
    actionable = latest[latest['Signal'] == 1].sort_values(by='Probability', ascending=False)
    
    print("\n" + "="*45)
    print(f"🚀 HIGH CONVICTION SETUPS (Threshold: {threshold:.3f})")
    print("="*45)
    
    if actionable.empty:
        print("No setups passed the strict precision threshold today.")
        print(f"Current Market Breadth: {latest['Market_Breadth'].iloc[0]:.1%} stocks above 20-EMA")
    else:
        print(f"Current Market Breadth: {latest['Market_Breadth'].iloc[0]:.1%} stocks above 20-EMA\n")
        print(actionable[['Ticker', 'Close', 'Probability', 'ATR_14']].head(10).to_string(index=False))

if __name__ == "__main__":
    screen_today()
