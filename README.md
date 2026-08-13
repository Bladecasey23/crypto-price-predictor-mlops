# Real-Time Bitcoin Price Direction Predictor

A real-time machine learning service that predicts whether Bitcoin's price will rise or fall in the next hour, served through a live API, containerized with Docker, and tested with CI.

This project focuses on **ML engineering** as much as modeling: live data ingestion, a served prediction API, automated testing, and containerized deployment — not just a notebook.

## Live Demo

**API is deployed and publicly accessible at:**
[https://btc-price-predictor-n2ga.onrender.com](https://btc-price-predictor-n2ga.onrender.com)

Try the live prediction endpoint: [https://btc-price-predictor-n2ga.onrender.com/predict](https://btc-price-predictor-n2ga.onrender.com/predict)

Interactive API docs: [https://btc-price-predictor-n2ga.onrender.com/docs](https://btc-price-predictor-n2ga.onrender.com/docs)

> **Note:** hosted on Render's free tier, which spins down after 15 minutes of inactivity. The first request after a period of idle time may take 30–60 seconds to respond while the service wakes up — subsequent requests are fast.

## Problem Statement

Short-term cryptocurrency price movement is notoriously difficult to predict — even professional quantitative trading firms with vastly more data and compute typically achieve only a small edge over random chance. This project doesn't attempt to "beat the market." Instead, it demonstrates a complete, honest, real-time ML pipeline: pulling live price data, engineering time-series features, training and evaluating a classifier, and serving live predictions through a deployed API.

## Data

- **Source:** [CoinGecko API](https://www.coingecko.com/en/api) — free, demo API key required
- **Asset:** Bitcoin (BTC), priced in USD
- **Granularity:** Hourly price data, ~90 days of history for training (~2,160 data points)

## Feature Engineering

Since raw price alone has little predictive value, the following features were engineered from the price history:

| Feature | Description |
|---|---|
| `pct_change_1h` | Percent price change from the previous hour |
| `rolling_mean_6h` | 6-hour rolling average price (short-term trend) |
| `rolling_mean_24h` | 24-hour rolling average price (longer-term trend) |
| `price_vs_24h_avg` | Current price relative to its 24h average (momentum signal) |
| `volatility_6h` | 6-hour rolling standard deviation (recent choppiness) |

**Target:** Binary — whether price is higher (`1`) or not (`0`) one hour later.

Rows with incomplete rolling windows (the first 24 hours) and the final row (no future price available yet) were dropped, reducing ~2,161 raw rows to ~2,137 usable rows.

## Model

- **Algorithm:** Random Forest Classifier (100 trees)
- **Train/test split:** Chronological 80/20 split — **not** randomly shuffled, since shuffling would let the model train on data "from the future" relative to its test set, which is unrealistic for a time-series problem.

### Results

| Metric | Value |
|---|---|
| Accuracy | 55.4% |
| Precision | 54.2% |
| Recall | 49.8% |
| F1 | 51.9% |

**Honest interpretation:** these results are only modestly better than random guessing (50%). Given how difficult short-term crypto price direction is to predict — a problem where even professional traders struggle for a small statistical edge — this is a credible, expected result rather than a shortcoming to hide. The value of this project lies in the engineering pipeline built around this problem, not in claiming predictive power the data doesn't support.

## Architecture

```
Live price data (CoinGecko) 
        ↓
Feature engineering (features_lib.py)
        ↓
Trained model (btc_model.pkl)
        ↓
FastAPI service (app.py) → /predict endpoint
        ↓
Dockerized for consistent deployment anywhere
```

## API

Built with **FastAPI**. Key endpoint:

- `GET /predict` — fetches the latest BTC price data, computes live features, and returns a prediction:
```json
{
  "timestamp": "2026-08-08T13:56:10.000",
  "current_price": 64972.29,
  "prediction": "down",
  "confidence": 0.58
}
```

Interactive API docs auto-generated at `/docs`.

## Testing

Unit tests (pytest) verify the feature engineering logic independently of live data — using small, hand-calculable synthetic datasets to confirm rolling averages, momentum, and column generation behave correctly.

```
pytest
```

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Running with Docker

```bash
docker build -t btc-predictor .
docker run -p 8000:8000 btc-predictor
```

## Tech Stack

Python, pandas, scikit-learn, FastAPI, uvicorn, joblib, pytest, Docker

## Limitations

- Predictive accuracy is close to random chance — expected given the inherent difficulty of short-term price prediction, but a genuine limitation nonetheless.
- Only price-based features are used; no order book data, sentiment data, or macroeconomic indicators, all of which likely carry additional signal.
- Model is trained on a single asset (BTC) and a fixed prediction horizon (1 hour); results may not generalize to other assets or timeframes.

## Possible Extensions

- Add more features (RSI, MACD, hour-of-day/day-of-week seasonality)
- Retrain periodically on a rolling data window
- Build a lightweight dashboard showing live price vs. live prediction over time
- Upgrade to a paid tier to eliminate cold-start delays
