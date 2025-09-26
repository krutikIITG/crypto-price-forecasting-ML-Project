import pandas as pd
import numpy as np
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

N = 2441

X = df['Open'].values.astype('float32')
Xtrain = X[:N]
Xtest = X[-272:]

Y = df['Mean'].values.astype('float32')
ytrain = Y[:N]
ytest = Y[-272:]
arr = ytest

from sklearn.linear_model import ElasticNet

for l1 in [0.1, 0.5, 0.9]:
    reg = ElasticNet(l1_ratio=l1, random_state=None)
    reg.fit(Xtrain.reshape(-1, 1), ytrain)
    ypred = reg.predict(Xtest.reshape(-1, 1))
    ytest_reshaped = ytest.reshape(-1, 1)

    plt.plot(arr, label='actual')
    plt.plot(ypred, label='predicted')
    plt.legend()
    plt.show()

    mse = np.mean((ypred.reshape(-1, 1) - ytest_reshaped) ** 2)
    rmse = np.sqrt(mse) + 201
    print("RMSE:", rmse)

    print("LINEAR REGRESSION with ELASTIC NET")
    print("Mean value depending on open")
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "365 day ago UTC")