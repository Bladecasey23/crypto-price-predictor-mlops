import requests
import pandas as pd

# The URL for CoinGecko's "market chart" endpoint for Bitcoin
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'

# Parameters we're sending with our request:
# vs_currency: what currency to price it in (USD)
# days: how many days of history to pull (90)
# interval: how granular the data points are (hourly)
params = {
    'vs_currency': 'usd',
    'days': '90',
    'interval': 'hourly'
}

# Make the actual request to CoinGecko's server
response = requests.get(url, params=params)

# Convert the response (which comes back as JSON text) into a Python dictionary
data = response.json()

# data['prices'] is a list of [timestamp_in_milliseconds, price] pairs
prices = data['prices']

# Turn that list into a proper pandas table with two columns
df = pd.DataFrame(prices, columns=['timestamp', 'price'])

# The timestamp is currently a raw number (milliseconds since 1970)  convert it to an actual readable date/time
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Let's look at what we got
print(df.shape)
print(df.head())
print(df.tail())

# Save it to a CSV so we don't have to re-fetch every time we work on it
df.to_csv('btc_prices.csv', index=False)
print("Saved to btc_prices.csv")