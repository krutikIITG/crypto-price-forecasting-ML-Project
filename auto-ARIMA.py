from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings("ignore")

print("Starting Auto-ARIMA model with manual parameter tuning...")

# Binance API key aur secret
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'

client = Client(api_key, api_secret)

# Cryptocurrency symbol
symbol = 'BTCUSDT'

print("Fetching data from Binance API...")
# Historical data fetch karna
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "500 day ago UTC")

print("Data loaded successfully from Binance API")

# Data ko DataFrame me convert karen
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

# Convert timestamps to readable date
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

# Convert price columns to numeric
df['low'] = pd.to_numeric(df['low'])
df['high'] = pd.to_numeric(df['high'])
df['open'] = pd.to_numeric(df['open'])
df['close'] = pd.to_numeric(df['close'])
df['volume'] = pd.to_numeric(df['volume'])
df['mean'] = (df['low'] + df['high']) / 2

# Clean data
df = df[['open_time', 'low', 'high', 'open', 'close', 'volume', 'mean']].dropna()
df.rename(columns={'open_time': 'Date'}, inplace=True)

print(f"Data shape: {df.shape}")

train_size = int(len(df) * 0.9)

# Prepare prediction dataset
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

# Manual Auto-ARIMA: Try different parameter combinations
print("Finding optimal ARIMA parameters manually...")

best_aic = float('inf')
best_order = None
best_model = None

# Parameter combinations to try
param_combinations = [
    (0,1,0), (0,1,1), (0,1,2),
    (1,1,0), (1,1,1), (1,1,2),
    (2,1,0), (2,1,1), (2,1,2),
    (3,1,0), (3,1,1), (3,1,2)
]

for order in param_combinations:
    try:
        model = ARIMA(train_y, order=order)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_order = order
            best_model = model_fit
        print(f"ARIMA{order} - AIC: {model_fit.aic:.2f}")
    except:
        continue

print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")

# Generate predictions using best model
print("Generating predictions...")
predictions = best_model.predict(start=len(train_y), end=len(train_y) + len(test_y) - 1)
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
plt.title(f'Auto-ARIMA {best_order} Bitcoin Price Prediction')
plt.show()

# Calculate RMSE
from statsmodels.tools.eval_measures import rmse
error = rmse(predicted_prices, actual_prices)
print("RMSE:", error)
print("Manual Auto-ARIMA model completed successfully!")
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")