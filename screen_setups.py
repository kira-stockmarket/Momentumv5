import pandas as pd
import joblib
from train_model import engineer_features

def screen_today():
    model = joblib.load("breakout_lgbm_model.pkl")
    df = pd.read_parquet("nifty500_ohlcv.parquet")
    data = engineer_features(df)
    
    features = ['Trend_Score', 'Dist_to_High', 'Vol_Ratio', 'ROC_20', 'Range_Ratio']
    
    # Get the latest trading day per ticker
    latest = data.sort_values('Date').groupby('Ticker').last().reset_index()
    latest['Probability'] = model.predict_proba(latest[features])[:, 1]
    
    # Top 5 candidates matching 20%+ setup
    top_setups = latest.sort_values(by='Probability', ascending=False)[['Ticker', 'Close', 'Probability']].head(5)
    print("\nTop 20%+ Momentum Setups:")
    print(top_setups.to_string(index=False))

if __name__ == "__main__":
    screen_today()
