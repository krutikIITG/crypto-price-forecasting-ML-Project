
from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

print("Starting Random Forest model with Binance API...")

# Binance API setup
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'
client = Client(api_key, api_secret)
symbol = 'BTCUSDT'

klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")

# Data preparation (same as XGBoost)
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
df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).apply(lambda x: x[x>0].mean() / abs(x[x<0].mean()))))
df['volatility'] = df['close'].rolling(window=10).std()

feature_cols = ['open', 'high', 'low', 'volume', 'sma_5', 'sma_20', 'rsi', 'volatility']
df_clean = df[feature_cols + ['close']].dropna()

X = df_clean[feature_cols]
y = df_clean['close']

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Random Forest model
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Visualization
plt.figure(figsize=(20,10))
plt.plot(range(len(y_test)), y_test.values, label='Actual', color='red')
plt.plot(range(len(predictions)), predictions, label='Random Forest Predicted', color='blue')
plt.legend()
plt.title('Random Forest Bitcoin Price Prediction')
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Random Forest RMSE: {rmse}")
print("Random Forest model completed successfully!")

