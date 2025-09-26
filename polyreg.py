import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import warnings
import requests
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

print("Starting Polynomial Regression model...")

# Binance API data fetch
url = "https://api.binance.com/api/v3/klines"
param = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": 1367107200000,
    "endTime": 1601510400000
}
response = requests.get(url, params=param)
data = response.json()

print("Data loaded successfully from Binance")

df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                 'Close time', 'Quote asset volume', 'Number of trades',
                                 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
df['Low'] = pd.to_numeric(df['Low'])
df['High'] = pd.to_numeric(df['High'])
df['Open'] = pd.to_numeric(df['Open'])
df['Close'] = pd.to_numeric(df['Close'])
df['Volume'] = pd.to_numeric(df['Volume'])
df['Mean'] = (df['Low'] + df['High']) / 2

df = df[['Date', 'Low', 'High', 'Open', 'Close', 'Volume', 'Mean']].dropna()

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size

dataset_for_prediction = df.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['Mean'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

sc_in = MinMaxScaler(feature_range=(0,1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0:'Low',1:'High',2:'Open',3:'Close',4:'Volume',5:'Mean'}, inplace=True)

sc_out = MinMaxScaler(feature_range=(0,1))
scaled_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaled_output = pd.DataFrame(scaled_output, index=dataset_for_prediction.index)
y = scaled_output
y.rename(columns={0:'BTC Price next day'}, inplace=True)

train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

print(f"Train data shape: {train_X.shape}, Test data shape: {test_X.shape}")

# Polynomial Regression Pipeline
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

print("Training Polynomial Regression model...")
poly_model.fit(train_X, train_y)

print("Generating predictions...")
predictions = poly_model.predict(test_X)
predictions = pd.DataFrame(predictions, columns=['Pred'], index=test_X.index)

act = pd.DataFrame(scaled_output.iloc[len(train_X):, 0])
predictions['Actual'] = act['BTC Price next day']

predicted_prices = sc_out.inverse_transform(predictions[['Pred']])
actual_prices = sc_out.inverse_transform(predictions[['Actual']])

plt.figure(figsize=(20,10))
plt.plot(predictions.index, predicted_prices, label='Predicted', color='blue')
plt.plot(predictions.index, actual_prices, label='Actual', color='red')
plt.legend()
plt.show()

from statsmodels.tools.eval_measures import rmse
error = rmse(predicted_prices, actual_prices)
print("RMSE:", error)
print("Polynomial Regression model completed successfully!")
from binance.client import Client
api_key = 'ADD_YOUR_BINANCE_API_KEY_HERE'
api_secret = 'ADD_YOUR_BINANCE_API_SECRET_HERE'
client = Client(api_key, api_secret)
klines = client.get_historical_klines('BTCUSDT', Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")