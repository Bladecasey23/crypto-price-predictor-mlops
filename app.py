from fastapi import FastAPI, HTTPException
import pandas as pd
import requests
import joblib
from features_lib import build_features, FEATURE_COLUMNS
from fastapi.responses import FileResponse
import time
import threading

# Simple in-memory cache
_cache = {"data": None, "timestamp": 0}
CACHE_DURATION_SECONDS = 300  # how often the background loop refreshes (5 min)
_cache_lock = threading.Lock()

# Create the FastAPI application
app = FastAPI(title="BTC Price Direction Predictor")

# Load the trained model once when the server starts
model = joblib.load("btc_model.pkl")


def fetch_recent_prices():
    """
    Pulls the last few days of hourly BTC prices from CoinGecko.
    We need at least 24+ hours of history because our features
    include a 24-hour rolling average.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "3",
        "interval": "hourly"
    }
    headers = {
        "User-Agent": "BTC-Price-Predictor/1.0"
    }
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )
    # CoinGecko rate limit
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="CoinGecko rate limit reached. Please try again shortly."
        )
    # Other CoinGecko errors
    if response.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Price data provider unavailable (status {response.status_code})."
        )
    data = response.json()
    if "prices" not in data:
        raise HTTPException(
            status_code=503,
            detail="Price data provider returned an unexpected response."
        )
    prices = data["prices"]
    df = pd.DataFrame(
        prices,
        columns=["timestamp", "price"]
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )
    return df


def compute_prediction():
    """
    Does the actual fetch + feature build + predict.
    Called by the startup event and the background refresh loop —
    never called directly from the /predict route anymore.
    """
    df = fetch_recent_prices()
    df = build_features(df)
    df_clean = df.dropna().reset_index(drop=True)
    latest_row = df_clean.iloc[[-1]]
    X_latest = latest_row[FEATURE_COLUMNS]

    prediction = model.predict(X_latest)[0]
    probability = model.predict_proba(X_latest)[0]

    return {
        "timestamp": str(
            latest_row["timestamp"].values[0]
        ),
        "current_price": float(
            latest_row["price"].values[0]
        ),
        "prediction": (
            "up" if prediction == 1 else "down"
        ),
        "confidence": float(max(probability))
    }


def refresh_loop():
    """
    Runs forever in a background thread, refreshing the cache
    every CACHE_DURATION_SECONDS. Never raises — a failed refresh
    just keeps serving the last good cached result.
    """
    while True:
        time.sleep(CACHE_DURATION_SECONDS)
        try:
            result = compute_prediction()
            with _cache_lock:
                _cache["data"] = result
                _cache["timestamp"] = time.time()
        except Exception as e:
            print(f"Background refresh failed: {e}")


@app.on_event("startup")
def start_background_refresh():
    # Do one synchronous fetch immediately so /predict has data
    # right away, then hand off to the background loop for future refreshes
    try:
        result = compute_prediction()
        with _cache_lock:
            _cache["data"] = result
            _cache["timestamp"] = time.time()
    except Exception as e:
        print(f"Initial fetch failed: {e}")

    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/predict")
def predict():
    with _cache_lock:
        if _cache["data"] is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction not ready yet, try again shortly."
            )
        return _cache["data"]
