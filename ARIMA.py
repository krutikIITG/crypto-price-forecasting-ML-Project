import pandas as pd
import matplotlib.pyplot as plt
import requests
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
warnings.filterwarnings("ignore")

url = "https://api.binance.com/api/v3/klines"
param = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": 1367107200000,
    "endTime": 1601510400000
}
response = requests.get(url, params=param)
data = response.json()

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

dataset_for_prediction = df.copy()
dataset_for_prediction['Actual'] = dataset_for_prediction['Mean'].shift()
dataset_for_prediction = dataset_for_prediction.dropna()
dataset_for_prediction['Date'] = pd.to_datetime(dataset_for_prediction['Date'])
dataset_for_prediction.index = dataset_for_prediction['Date']

from sklearn.preprocessing import MinMaxScaler
sc_in = MinMaxScaler(feature_range=(0, 1))
scaled_input = sc_in.fit_transform(dataset_for_prediction[['Low', 'High', 'Open', 'Close', 'Volume', 'Mean']])
scaled_input = pd.DataFrame(scaled_input, index=dataset_for_prediction.index)
X = scaled_input
X.rename(columns={0:'Low', 1:'High', 2:'Open', 3:'Close', 4:'Volume', 5:'Mean'}, inplace=True)

sc_out = MinMaxScaler(feature_range=(0, 1))
scaled_output = sc_out.fit_transform(dataset_for_prediction[['Actual']])
scaled_output = pd.DataFrame(scaled_output, index=dataset_for_prediction.index)
y = scaled_output
y.rename(columns={0:'BTC Price next day'}, inplace=True)
y.index = dataset_for_prediction.index

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()
test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_y, exog=train_X, order=(0, 1, 1))
results = model.fit()

# Train/test data split ke baad
print(f"train_X shape: {train_X.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_X shape: {test_X.shape}")
print(f"test_y shape: {test_y.shape}")



# Predict
predictions = results.predict(start=len(train_X), end=len(train_X) + len(test_X) - 1, exog=test_X)


print(f"Predictions length: {len(predictions)}")
print(predictions.head())

if len(predictions) > 0:
    predictions_df = pd.DataFrame(predictions.values, columns=['Pred'])
    predictions_df.index = test_X.index
    predictions = predictions_df
else:
    print("No predictions generated!")


act = pd.DataFrame(scaled_output.iloc[len(train_X):, 0])
predictions['Actual'] = act['BTC Price next day']

testPredict = sc_out.inverse_transform(predictions[['Pred']])
testActual = sc_out.inverse_transform(predictions[['Actual']])


plt.figure(figsize=(20,10))
plt.plot(predictions.index, testActual, label='Actual', color='blue')
plt.plot(predictions.index, testPredict, label='Predicted', color='red')
plt.legend()
plt.show()

from statsmodels.tools.eval_measures import rmse
print("RMSE:", rmse(testActual, testPredict))

