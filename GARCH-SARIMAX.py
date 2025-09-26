import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
import arch
register_matplotlib_converters()
import warnings
import requests
warnings.filterwarnings("ignore")

print("Starting GARCH-SARIMAX model...")

url = "https://api.binance.com/api/v3/klines"
param = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": 1367107200000,
    "endTime": 1601510400000
}
response = requests.get(url, params=param)
data = response.json()

print("Data loaded successfully")

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

from sklearn.preprocessing import MinMaxScaler
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
y.index = dataset_for_prediction.index

train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

print(f"Train data shape: {train_X.shape}, Test data shape: {test_X.shape}")

from statsmodels.tsa.statespace.sarimax import SARIMAX

print("Model training started...")

predic_garch = []
for i in range(len(test_X)):
    try:
        model = SARIMAX(pd.concat([train_y, test_y.iloc[:i+1]]),
                        exog=pd.concat([train_X, test_X.iloc[:i+1]]),
                        order=(0,1,1),
                        seasonal_order=(0,0,1,12),
                        enforce_invertibility=False,
                        enforce_stationarity=False)
        results = model.fit(disp=False)
        garch = arch.arch_model(results.resid, p=1, q=1, vol='GARCH')
        garch_model = garch.fit(update_freq=0, disp='off')
        garch_forecast = garch_model.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]
        predic_garch.append(predicted_et)
    except:
        predic_garch.append(0)  # Fallback if GARCH fails

model = SARIMAX(train_y,
                exog=train_X,
                order=(0,1,1),
                seasonal_order=(0,0,1,12),
                enforce_invertibility=False,
                enforce_stationarity=False)
results = model.fit(disp=False)

print("Generating predictions...")

# Fixed prediction generation
predictions = results.predict(start=len(train_X), end=len(train_X) + len(test_X) - 1, exog=test_X)

# Convert to DataFrame properly
predictions = pd.DataFrame(predictions.values, columns=['Pred'], index=test_X.index)

# Add GARCH component
for i in range(len(predictions)):
    if i < len(predic_garch):
        predictions.iloc[i, 0] += predic_garch[i]

# Add actual values
act = pd.DataFrame(scaled_output.iloc[len(train_X):, 0])
predictions['Actual'] = act['BTC Price next day']

print(f"Predictions shape: {predictions.shape}")

# Transform back to original scale
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
print("GARCH-SARIMAX model completed successfully!")
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")