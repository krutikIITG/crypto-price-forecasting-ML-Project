from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR
warnings.filterwarnings("ignore")

print("Starting VAR model with Binance API...")

# Binance API key aur secret
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'

client = Client(api_key, api_secret)

# Cryptocurrency symbol
symbol = 'BTCUSDT'

print("Fetching data from Binance API...")
# Historical data fetch karna
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "400 day ago UTC")

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

# Prepare multivariate dataset for VAR
dataset_for_prediction = df.copy()
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

# Select multiple variables for VAR (multivariate analysis)
var_data = dataset_for_prediction[['close', 'volume', 'high', 'low']].dropna()

# Scale data
sc = MinMaxScaler(feature_range=(0,1))
scaled_data = sc.fit_transform(var_data)
scaled_df = pd.DataFrame(scaled_data, 
                        columns=['close', 'volume', 'high', 'low'],
                        index=var_data.index)

print(f"VAR data shape: {scaled_df.shape}")

# Train/test split
train_data = scaled_df[:train_size]
test_data = scaled_df[train_size:]

print(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

# VAR Model
print("Training VAR model...")
try:
    # Create VAR model
    model = VAR(train_data)
    
    # Find optimal lag order (Fixed: removed verbose parameter)
    print("Finding optimal lag order...")
    lag_order_results = model.select_order(maxlags=8)
    optimal_lags = lag_order_results.aic
    print(f"Optimal lag order (AIC): {optimal_lags}")
    
    # Fit model with optimal lags
    print("Fitting VAR model...")
    var_model = model.fit(optimal_lags)
    
    print("Generating predictions...")
    # Forecast
    forecast_steps = len(test_data)
    forecast_input = train_data.values[-optimal_lags:]
    forecast = var_model.forecast(forecast_input, steps=forecast_steps)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(forecast, 
                              columns=['close', 'volume', 'high', 'low'],
                              index=test_data.index)
    
    print(f"Forecast shape: {forecast_df.shape}")
    
    # Focus on 'close' price predictions
    predicted_close = forecast_df['close'].values.reshape(-1, 1)
    actual_close = test_data['close'].values.reshape(-1, 1)
    
    # Create separate scaler for close prices
    sc_close = MinMaxScaler(feature_range=(0,1))
    sc_close.fit(var_data[['close']])
    
    # Transform back to original scale
    predicted_prices = sc_close.inverse_transform(predicted_close)
    actual_prices = sc_close.inverse_transform(actual_close)
    
    # Plot results
    plt.figure(figsize=(20,10))
    plt.plot(test_data.index, predicted_prices, label='VAR Predicted', color='blue')
    plt.plot(test_data.index, actual_prices, label='Actual', color='red')
    plt.legend()
    plt.title('VAR Bitcoin Close Price Prediction')
    plt.show()
    
    # Calculate RMSE
    from statsmodels.tools.eval_measures import rmse
    error = rmse(predicted_prices, actual_prices)
    print("RMSE:", error)
    print("VAR model completed successfully!")
    
    # Print model summary
    print(f"\nVAR Model Details:")
    print(f"Variables: {list(train_data.columns)}")
    print(f"Optimal lags: {optimal_lags}")
    print(f"Forecast periods: {forecast_steps}")
    
except Exception as e:
    print(f"Error in VAR model: {e}")
    print("VAR model requires sufficient data and may need parameter adjustment")

klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")
