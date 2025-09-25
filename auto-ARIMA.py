from binance.client import Client
import pandas as pd

# Binance API key aur secret yahan daalen (jo aapne banaya hai)
api_key = 'QMIckrMdKQQDhhoXmB5GypBHSDtCS4EpYGVikeHQWBd4aaHd2u9VxOpWySNFHTV5'
api_secret = 'FiA4y1CsnoRmUyS0nsdYYgL9HPSq2Umc35BEHiz2Dd7cAws74qG4G44FG6sxXfmT'

client = Client(api_key, api_secret)

# Koi cryptocurrency symbol, jaise BTCUSDT (Bitcoin vs USDT)
symbol = 'BTCUSDT'

# Historical data fetch karna (klines)
# interval: '1d' daily data, limit=100 (last 100 days)
klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "100 day ago UTC")

# Data ko DataFrame me convert karen
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
    'taker_buy_quote_asset_volume', 'ignore'
])

# Convert timestamps to readable date
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

print(df.head())
