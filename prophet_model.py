
from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

print("Starting Prophet model with Binance API...")

api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'
client = Client(api_key, api_secret)
symbol = 'BTCUSDT'

klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

# Prophet format
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

df['ds'] = pd.to_datetime(df['open_time'], unit='ms')
df['y'] = pd.to_numeric(df['close'])
prophet_df = df[['ds', 'y']]

# Train/test split
train_size = int(len(prophet_df) * 0.8)
train_df = prophet_df[:train_size]
test_df = prophet_df[train_size:]

# Prophet model
print("Training Prophet model...")
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(train_df)

# Predictions
future = model.make_future_dataframe(periods=len(test_df))
forecast = model.predict(future)

# Plot
plt.figure(figsize=(20,10))
plt.plot(test_df['ds'], test_df['y'], label='Actual', color='red')
plt.plot(forecast['ds'].iloc[train_size:], forecast['yhat'].iloc[train_size:], label='Prophet Predicted', color='blue')
plt.legend()
plt.title('Prophet Bitcoin Price Prediction')
plt.show()

print("Prophet model completed successfully!")

