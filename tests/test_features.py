import pandas as pd
from features_lib import build_features, FEATURE_COLUMNS


def test_build_features_adds_expected_columns():
    """
    Checks that build_features() actually creates all the columns we expect.
    If someone accidentally deletes a feature or renames one, this test will catch it.
    """
    # Create a small fake dataset we don't need real BTC data to test the LOGIC
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=30, freq='h'),
        'price': [100 + i for i in range(30)]  # a simple rising price sequence
    })

    result = build_features(sample_data)

    for col in FEATURE_COLUMNS:
        assert col in result.columns


def test_build_features_rolling_mean_is_correct():
    """
    Checks that the 6-hour rolling mean is actually calculated correctly,
    using a simple sequence where we can compute the expected answer by hand.
    """
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=10, freq='h'),
        'price': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })

    result = build_features(sample_data)

    # The 6-hour rolling mean at row index 5 (7th row) should be the average of rows 0-5:
    # (10+20+30+40+50+60)/6 = 35
    assert result.loc[5, 'rolling_mean_6h'] == 35


def test_price_vs_24h_avg_is_zero_when_price_equals_average():
    """
    Sanity check: if price exactly equals its own rolling average,
    price_vs_24h_avg should be 0 (no deviation).
    """
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=30, freq='h'),
        'price': [100] * 30  # constant price, never changes
    })

    result = build_features(sample_data)

    # once we have enough history, price should exactly match its own rolling average
    assert abs(result.loc[29, 'price_vs_24h_avg']) < 1e-9