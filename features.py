import pandas as pd

# Load the price data we saved earlier
df = pd.read_csv('btc_prices.csv', parse_dates=['timestamp'])

# Sort by time just to be safe (APIs usually return in order, but don't assume)
df = df.sort_values('timestamp').reset_index(drop=True)

# --- Feature 1: Percent change from the previous hour ---
# This tells the model: "was the last hour's move up or down, and by how much?"
df['pct_change_1h'] = df['price'].pct_change(1)

# --- Feature 2: Rolling average over the last 6 hours ---
# Smooths out noise, shows the short-term trend direction
df['rolling_mean_6h'] = df['price'].rolling(window=6).mean()

# --- Feature 3: Rolling average over the last 24 hours ---
# Shows the longer-term trend, so the model can compare short vs long trend
df['rolling_mean_24h'] = df['price'].rolling(window=24).mean()

# --- Feature 4: Price relative to its 24h rolling average ---
# If price is above its recent average, that might signal momentum
df['price_vs_24h_avg'] = df['price'] / df['rolling_mean_24h'] - 1

# --- Feature 5: Rolling volatility (standard deviation) over 6 hours ---
# Captures how "choppy" recent price action has been
df['volatility_6h'] = df['price'].rolling(window=6).std()

# --- The target: will price be higher 1 hour from now? ---
# We shift price backward by 1 so each row can "see into the future" by one step
df['future_price'] = df['price'].shift(-1)
df['target'] = (df['future_price'] > df['price']).astype(int)  # 1 = price went up, 0 = went down/same

print(df.head(10))
print(df.tail(10))
print(df.shape)

df.to_csv('btc_features.csv', index=False)
print("Saved to btc_features.csv")

df = pd.read_csv('btc_features.csv', parse_dates=['timestamp'])

# Drop rows with any NaN values  this removes the first 24 rows (rolling window warm-up)
# and the very last row (no "future" price to compare against)
df_clean = df.dropna().reset_index(drop=True)

print(df_clean.shape)
print(df_clean.isna().sum())

df_clean.to_csv('btc_features_clean.csv', index=False)
print("Saved to btc_features_clean.csv")