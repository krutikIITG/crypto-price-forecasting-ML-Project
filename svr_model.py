
from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

print("Starting SVR model with Binance API...")

api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'
client = Client(api_key, api_secret)
symbol = 'BTCUSDT'

klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

# Same data prep as other ML models
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

df['open'] = pd.to_numeric(df['open'])
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['close'] = pd.to_numeric(df['close'])
df['volume'] = pd.to_numeric(df['volume'])

df['sma_5'] = df['close'].rolling(window=5).mean()
df['sma_20'] = df['close'].rolling(window=20).mean()
df['volatility'] = df['close'].rolling(window=10).std()

feature_cols = ['open', 'high', 'low', 'volume', 'sma_5', 'sma_20', 'volatility']
df_clean = df[feature_cols + ['close']].dropna()

X = df_clean[feature_cols]
y = df_clean['close']

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# SVR model
print("Training SVR model...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

model = SVR(kernel='rbf', C=1000, gamma=0.01)
model.fit(X_train_scaled, y_train_scaled)

predictions_scaled = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

# Visualization
plt.figure(figsize=(20,10))
plt.plot(range(len(y_test)), y_test.values, label='Actual', color='red')
plt.plot(range(len(predictions)), predictions, label='SVR Predicted', color='blue')
plt.legend()
plt.title('SVR Bitcoin Price Prediction')
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"SVR RMSE: {rmse}")
print("SVR model completed successfully!")

