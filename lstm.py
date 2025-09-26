
from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

print("Starting LSTM model with Binance API...")

# Binance API setup
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'
client = Client(api_key, api_secret)
symbol = 'BTCUSDT'

print("Fetching Bitcoin data from Binance...")
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

# Data preprocessing
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

df['close'] = pd.to_numeric(df['close'])
prices = df['close'].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Create sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Train/test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

print("Training LSTM model...")
model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

# Predictions
predictions = model.predict(X_test)

# Scale back
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_scaled = scaler.inverse_transform(predictions)

# Plot results
plt.figure(figsize=(20,10))
plt.plot(range(len(y_test_scaled)), y_test_scaled, label='Actual', color='red')
plt.plot(range(len(predictions_scaled)), predictions_scaled, label='LSTM Predicted', color='blue')
plt.legend()
plt.title('LSTM Bitcoin Price Prediction')
plt.show()

# RMSE
rmse = np.sqrt(np.mean((y_test_scaled - predictions_scaled)**2))
print(f"LSTM RMSE: {rmse}")
print("LSTM model completed successfully!")

