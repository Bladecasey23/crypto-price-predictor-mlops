import pandas as pd

def build_features(df):
    """
    Takes a DataFrame with 'timestamp' and 'price' columns,
    returns it with engineered features added.
    Used both for training and for live predictions, so the logic
    stays identical in both places.
    """
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['pct_change_1h'] = df['price'].pct_change(1)
    df['rolling_mean_6h'] = df['price'].rolling(window=6).mean()
    df['rolling_mean_24h'] = df['price'].rolling(window=24).mean()
    df['price_vs_24h_avg'] = df['price'] / df['rolling_mean_24h'] - 1
    df['volatility_6h'] = df['price'].rolling(window=6).std()

    return df

FEATURE_COLUMNS = ['pct_change_1h', 'rolling_mean_6h', 'rolling_mean_24h', 'price_vs_24h_avg', 'volatility_6h']