from fastapi import FastAPI
import pandas as pd
import requests
import joblib
from features_lib import build_features, FEATURE_COLUMNS
from fastapi.responses import FileResponse
from fastapi import HTTPException,FastAPI
import time

# simple in-memory cache
_cache = {"data": None, "timestamp": 0}
CACHE_DURATION_SECONDS = 900  # 15 minutes

# Create the FastAPI application object  this is what uvicorn will run
app = FastAPI(title="BTC Price Direction Predictor")

# Load the trained model once when the server starts, not on every request
# (loading from disk is slow  we don't want to do it every single time someone asks for a prediction)
model = joblib.load('btc_model.pkl')

def fetch_recent_prices():
    """
        Pulls the last few days of hourly BTC prices from CoinGecko.
        We need at least 24+ hours of history because our features
        include a 24-hour rolling average.
        """
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': '3', 'interval': 'hourly'}
    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Price data provider unavailable (status {response.status_code}). Please try again shortly."
        )

    data = response.json()

    if 'prices' not in data:
        raise HTTPException(
            status_code=503,
            detail="Price data provider returned an unexpected response. Please try again shortly."
        )

    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/predict")
def predict():
    now = time.time()

    # Return cached result if it's still fresh
    if _cache["data"] and (now - _cache["timestamp"] < CACHE_DURATION_SECONDS):
        return _cache["data"]

    df = fetch_recent_prices()
    df = build_features(df)
    df_clean = df.dropna().reset_index(drop=True)

    latest_row = df_clean.iloc[[-1]]
    X_latest = latest_row[FEATURE_COLUMNS]

    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]

    result = {
        "timestamp": str(latest_row['timestamp'].values[0]),
        "current_price": float(latest_row['price'].values[0]),
        "prediction": "up" if prediction == 1 else "down",
        "confidence": float(max(probability))
    }

    # Save to cache
    _cache["data"] = result
    _cache["timestamp"] = now

    return result