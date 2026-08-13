from fastapi.testclient import TestClient

import app as app_module
from app import app


client = TestClient(app)


def test_predict_returns_503_when_cache_is_empty():
    """
    /predict should never fall back to a live CoinGecko fetch anymore —
    if the background loop hasn't populated the cache yet, it should
    return 503 rather than hang on a network call.
    """
    # Force the cache into its "not ready yet" state
    app_module._cache["data"] = None
    app_module._cache["timestamp"] = 0

    response = client.get("/predict")

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()


def test_predict_returns_whatever_is_in_the_cache():
    """
    /predict should just serve _cache['data'] as-is, with no extra
    fetching, feature-building, or prediction logic in the route itself.
    """
    fake_result = {
        "timestamp": "2026-08-14T12:00:00",
        "current_price": 65000.0,
        "prediction": "up",
        "confidence": 0.61,
    }
    app_module._cache["data"] = fake_result
    app_module._cache["timestamp"] = 123456789.0

    response = client.get("/predict")

    assert response.status_code == 200
    assert response.json() == fake_result


def test_compute_prediction_builds_result_from_mocked_price_data(monkeypatch):
    """
    compute_prediction() should fetch prices, build features, run the
    model, and shape the result dict correctly — tested here with a
    mocked fetch_recent_prices() so no real network call is made and
    no real model prediction is required to match a specific value.
    """
    import pandas as pd

    sample_prices = pd.DataFrame({
        "timestamp": pd.date_range(start="2026-01-01", periods=48, freq="h"),
        "price": [100 + i for i in range(48)],
    })

    monkeypatch.setattr(app_module, "fetch_recent_prices", lambda: sample_prices)

    result = app_module.compute_prediction()

    assert set(result.keys()) == {
        "timestamp",
        "current_price",
        "prediction",
        "confidence",
    }
    assert result["prediction"] in ("up", "down")
    assert 0.0 <= result["confidence"] <= 1.0
