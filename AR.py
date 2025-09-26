from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

print("Starting ARIMA model with Binance API...")

# Binance API setup
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'
client = Client(api_key, api_secret)

# Cryptocurrency symbol
symbol = 'BTCUSDT'

print("Fetching Bitcoin data from Binance...")
# Standardized data fetch - last 365 days
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

print("Data loaded successfully from Binance API")

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

# Convert timestamps and prices
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['low'] = pd.to_numeric(df['low'])
df['high'] = pd.to_numeric(df['high'])
df['open'] = pd.to_numeric(df['open'])
df['close'] = pd.to_numeric(df['close'])
df['volume'] = pd.to_numeric(df['volume'])
df['mean'] = (df['low'] + df['high']) / 2

# Clean and rename
df = df[['open_time', 'low', 'high', 'open', 'close', 'volume', 'mean']].dropna()
df.rename(columns={'open_time': 'Date'}, inplace=True)

print("Script executed successfully")
print(df.head())
print(f"Data shape: {df.shape}")

# Prepare data for ARIMA
train_size = int(len(df) * 0.9)

dataset_for_prediction = df.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['mean'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# Scale data
sc_out = MinMaxScaler(feature_range=(0,1))
scaled_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaled_output = pd.DataFrame(scaled_output, index=dataset_for_prediction.index)
y = scaled_output
y.rename(columns={0:'BTC Price next day'}, inplace=True)

# Train/test split
train_y = y[:train_size].dropna()
test_y = y[train_size:].dropna()

print(f"Train data shape: {train_y.shape}, Test data shape: {test_y.shape}")

# ARIMA Model
print("Training ARIMA model...")
try:
    model = ARIMA(train_y, order=(1,1,1))
    arima_model = model.fit()
    
    print("Generating predictions...")
    predictions = arima_model.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1)
    predictions = pd.DataFrame(predictions.values, columns=['Pred'], index=test_y.index)
    
    # Add actual values
    act = pd.DataFrame(scaled_output.iloc[len(train_y):, 0])
    predictions['Actual'] = act['BTC Price next day']
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Transform back to original scale
    predicted_prices = sc_out.inverse_transform(predictions[['Pred']])
    actual_prices = sc_out.inverse_transform(predictions[['Actual']])
    
    # Plot results
    plt.figure(figsize=(20,10))
    plt.plot(predictions.index, predicted_prices, label='Predicted', color='blue')
    plt.plot(predictions.index, actual_prices, label='Actual', color='red')
    plt.legend()
    plt.title('ARIMA Bitcoin Price Prediction (365 Days)')
    plt.show()
    
    # Calculate RMSE
    from statsmodels.tools.eval_measures import rmse
    error = rmse(predicted_prices, actual_prices)
    print("RMSE:", error)
    print("ARIMA model completed successfully!")

except Exception as e:
    print(f"Error in ARIMA: {e}")
    print("Model training failed")
