from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

print("Starting SARIMAX model with Binance API...")

# Binance API key aur secret
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'

client = Client(api_key, api_secret)

# Cryptocurrency symbol
symbol = 'BTCUSDT'

print("Fetching data from Binance API...")
# Historical data fetch karna (1 year data)
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

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

# Scale input features (exogenous variables)
sc_in = MinMaxScaler(feature_range=(0,1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['low', 'high', 'open', 'close', 'volume', 'mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Mean'}, inplace=True)

# Scale output
sc_out = MinMaxScaler(feature_range=(0,1))
scaled_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaled_output = pd.DataFrame(scaled_output, index=dataset_for_prediction.index)
y = scaled_output
y.rename(columns={0:'BTC Price next day'}, inplace=True)

# Train/test split
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

print(f"Train data shape: {train_X.shape}, Test data shape: {test_X.shape}")

# SARIMAX Model
print("Training SARIMAX model...")
try:
    model = SARIMAX(train_y, 
                    exog=train_X,
                    order=(1, 1, 1),           # ARIMA part
                    seasonal_order=(1, 1, 1, 12),  # Seasonal part
                    enforce_invertibility=False,
                    enforce_stationarity=False)
    
    results = model.fit(disp=False)
    
    print("Generating predictions...")
    predictions = results.predict(start=len(train_X), 
                                 end=len(train_X) + len(test_X) - 1, 
                                 exog=test_X)
    
    predictions = pd.DataFrame(predictions.values, columns=['Pred'], index=test_X.index)
    
    # Add actual values
    act = pd.DataFrame(scaled_output.iloc[len(train_X):, 0])
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
    plt.title('SARIMAX Bitcoin Price Prediction')
    plt.show()
    
    # Calculate RMSE
    from statsmodels.tools.eval_measures import rmse
    error = rmse(predicted_prices, actual_prices)
    print("RMSE:", error)
    print("SARIMAX model completed successfully!")
    
except Exception as e:
    print(f"Error in SARIMAX: {e}")
    print("Trying simpler SARIMAX parameters...")
