import json
import requests
import threading
import pandas as pd
import numpy as np
import websocket
from datetime import datetime, timedelta
from fastquant import get_crypto_data
from binance.client import Client
import time
import yaml


def computeRSI(data, time_window):
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def stochastic(data, k_window, d_window, window):
    min_val = data.rolling(window=window, center=False).min()
    max_val = data.rolling(window=window, center=False).max()

    stoch = ((data - min_val) / (max_val - min_val)) * 100

    K = stoch.rolling(window=k_window, center=False).mean()

    D = K.rolling(window=d_window, center=False).mean()

    return K, D


def get_binance_bars(symbol, interval, startTime, endTime):
    url = "https://api.binance.com/api/v3/klines"
    startTime = str(int(startTime.timestamp() * 1000))
    endTime = str(int(endTime.timestamp() * 1000))
    limit = '1000'

    req_params = {"symbol": symbol, 'interval': interval, 'startTime': startTime, 'endTime': endTime, 'limit': limit}

    df = pd.DataFrame(json.loads(requests.get(url, params=req_params).text))

    if len(df.index) == 0:
        return None

    df = df.iloc[:, 0:6]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    df.open = df.open.astype("float")
    df.high = df.high.astype("float")
    df.low = df.low.astype("float")
    df.close = df.close.astype("float")
    df.volume = df.volume.astype("float")
    rsi = computeRSI(df.close, 14)
    df['stoch_rsi'], D = stochastic(rsi, 3, 3, 14)
    df['pvt_up'] = float(0)
    df['rsi_up_d'] = float(0)
    df['rsi_up_5'] = float(0)
    df['rsi_up_15'] = float(0)
    df['rsi_up_1H'] = float(0)
    df['volume'] = float(0)
    df['buy'] = float(0)

    return df


def on_message_1(ws, message):
    global one, closes_1
    message = json.loads(message)
    candle = message['k']
    if candle['x']:
        time_ = candle['T']
        opn = candle['o']
        high = candle['h']
        low = candle['l']
        close = candle['c']
        volume = candle['v']
        closes_1 = np.append(closes_1[:], float(close))
        t1m = threading.Thread(target=update_df_1, args=(time_, opn, high, low, close, volume,))
        t1m.start()


def on_message_5(ws, message):
    global five, closes_5
    message = json.loads(message)
    candle = message['k']
    if candle['x']:
        time_5 = candle['T']
        opn_5 = candle['o']
        high_5 = candle['h']
        low_5 = candle['l']
        close_5 = candle['c']
        volume_5 = candle['v']
        closes_5 = np.append(closes_5[:], float(close_5))
        t5m = threading.Thread(target=update_df_5, args=(time_5, opn_5, high_5, low_5, close_5, volume_5,))
        t5m.start()


def on_message_15(ws, message):
    global fifteen, closes_15
    message = json.loads(message)
    candle = message['k']
    if candle['x']:
        time_15 = candle['T']
        opn_15 = candle['o']
        high_15 = candle['h']
        low_15 = candle['l']
        close_15 = candle['c']
        volume_15 = candle['v']
        closes_15 = np.append(closes_15[:], float(close_15))
        t15m = threading.Thread(target=update_df_15, args=(time_15, opn_15, high_15, low_15, close_15, volume_15,))
        t15m.start()


def on_message_1H(ws, message):
    global hour, closes_1H
    message = json.loads(message)
    candle = message['k']
    if candle['x']:
        time_1H = candle['T']
        opn_1H = candle['o']
        high_1H = candle['h']
        low_1H = candle['l']
        close_1H = candle['c']
        volume_1H = candle['v']
        closes_1H = np.append(closes_1H[:], float(close_1H))
        t1H = threading.Thread(target=update_df_1H, args=(time_1H, opn_1H, high_1H, low_1H, close_1H, volume_1H,))
        t1H.start()


def on_message_1D(ws, message):
    global daily
    message = json.loads(message)
    candle = message['k']
    if candle['x']:
        t1D = threading.Thread(target=update_df_1D, args=())
        t1D.start()


def wsthread_1(one):
    ws = websocket.WebSocketApp(SOCKET_1, on_message=on_message_1)
    ws.run_forever()


def wsthread_5(five):
    ws = websocket.WebSocketApp(SOCKET_5, on_message=on_message_5)
    ws.run_forever()


def wsthread_15(fifteen):
    ws = websocket.WebSocketApp(SOCKET_15, on_message=on_message_15)
    ws.run_forever()


def wsthread_1H(hour):
    ws = websocket.WebSocketApp(SOCKET_1H, on_message=on_message_1H)
    ws.run_forever()


def wsthread_1D(daily):
    ws = websocket.WebSocketApp(SOCKET_1D, on_message=on_message_1D)
    ws.run_forever()


def update_df_1(time_, opn, high, low, close, volume):
    global one, five, fifteen, hour, daily, closes_1, cum_vol, order_size
    df1 = pd.DataFrame(list(closes_1[-35:]), columns={'close'})
    df1['rsi_14'] = computeRSI(df1.close, 14)
    df1['K'], df1['D'] = stochastic(df1['rsi_14'], 3, 3, 14)
    cum_vol += float(volume)
    pvt_bool_d = checkPVT(daily, close, cum_vol)
    rsi_bool_d = checkRSI(daily, close, 50)
    rsi_bool_5 = checkRSI_5(five, close)
    rsi_bool_15 = checkRSI(fifteen, close, 50)
    rsi_bool_1H = checkRSI(hour, close, 50)
    condition = condition_buy(pvt_bool_d, rsi_bool_d, rsi_bool_5, rsi_bool_15, rsi_bool_1H)
    if condition == 1:
        buy_quantity = order_size / float(close)
        buy_order = Client.create_test_order(
            symbol='BTCUSDT',
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=buy_quantity
        )
        print('Buy Order')
    last = pd.DataFrame({'datetime': [time_], 'open': [float(opn)], 'high': [float(high)], 'low': [float(low)],
                         'close': [float(close)], 'volume': [float(volume)], 'stoch_rsi': [float(df1['K'].values[-1])],
                         'pvt_up': [float(pvt_bool_d)], 'rsi_up_d': [float(rsi_bool_d)],
                         'rsi_up_5': [float(rsi_bool_5)], 'rsi_up_15': [float(rsi_bool_15)],
                         'rsi_up_1H': [float(rsi_bool_1H)], 'buy': [float(condition)]})
    one = one.append(last, ignore_index=True)
    print(f'{str(datetime.fromtimestamp(int(time_ / 1000)))} - Update: 1m - Cumulative Daily Volume: {int(cum_vol)}')


def update_df_5(time_5, opn_5, high_5, low_5, close_5, volume_5):
    global five, closes_5
    df5 = pd.DataFrame(list(closes_5[-35:]), columns={'close'})
    df5['rsi_14'] = computeRSI(df5.close, 14)
    df5['K'], df5['D'] = stochastic(df5['rsi_14'], 3, 3, 14)
    last = pd.DataFrame({'datetime': [time_5], 'open': [float(opn_5)], 'high': [float(high_5)], 'low': [float(low_5)],
                         'close': [float(close_5)], 'volume': [float(volume_5)],
                         'stoch_rsi': [float(df5['K'].values[-1])]})
    five = five.append(last, ignore_index=True)
    time.sleep(1)
    print(f'{str(datetime.fromtimestamp(int(time_5 / 1000)))} - Update: 5m')


def update_df_15(time_15, opn_15, high_15, low_15, close_15, volume_15):
    global fifteen, closes_15
    df15 = pd.DataFrame(list(closes_15[-35:]), columns={'close'})
    df15['rsi_14'] = computeRSI(df15.close, 14)
    df15['K'], df15['D'] = stochastic(df15['rsi_14'], 3, 3, 14)
    last = pd.DataFrame({'datetime': [time_15], 'open': [float(opn_15)], 'high': [float(high_15)],
                         'low': [float(low_15)], 'close': [float(close_15)], 'volume': [float(volume_15)],
                         'stoch_rsi': [float(df15['K'].values[-1])]})
    fifteen = fifteen.append(last, ignore_index=True)
    time.sleep(2)
    print(f'{str(datetime.fromtimestamp(int(time_15 / 1000)))} - Update: 15m')


def update_df_1H(time_1H, opn_1H, high_1H, low_1H, close_1H, volume_1H):
    global hour, closes_1H
    df1H = pd.DataFrame(list(closes_1H[-35:]), columns={'close'})
    df1H['rsi_14'] = computeRSI(df1H.close, 14)
    df1H['K'], df1H['D'] = stochastic(df1H['rsi_14'], 3, 3, 14)
    last = pd.DataFrame({'datetime': [time_1H], 'open': [float(opn_1H)], 'high': [float(high_1H)],
                         'low': [float(low_1H)], 'close': [float(close_1H)], 'volume': [float(volume_1H)],
                         'stoch_rsi': [float(df1H['K'].values[-1])]})
    hour = hour.append(last, ignore_index=True)
    time.sleep(3)
    print(f'{str(datetime.fromtimestamp(int(time_1H / 1000)))} - Update: 1h')


def update_df_1D():
    global daily, cum_vol
    daily = get_crypto_data("BTC/USDT", "2017-08-17", str(datetime.today() - timedelta(days=1)).split()[0])
    daily['x'] = ((daily.close - daily.close.shift(1)) / daily.close.shift(1)) * daily.volume
    cum_vol = 0
    time.sleep(4)
    print('------------- Update: 1d -------------')


def checkRSI(data, close, thresh):
    new_df = data.loc[:, ['close']].copy()
    new_row = pd.DataFrame({'close': [float(close)]})
    new_df = new_df.append(new_row, ignore_index=True)
    rsi = computeRSI(new_df.close, 14)
    new_df['stoch_rsi'], D = stochastic(rsi, 3, 3, 14)
    if (new_df['stoch_rsi'].values[-1] > thresh) & (new_df['stoch_rsi'].values[-1] >= new_df['stoch_rsi'].values[-2]):
        return 1
    else:
        return 0


def checkPVT(data, close, cum_vol):
    new_df = data.loc[:, ['close', 'volume']].copy()
    new_row = pd.DataFrame({'close': [float(close)], 'volume': [float(cum_vol)]})
    new_df = new_df.append(new_row, ignore_index=True)
    new_df['x'] = ((new_df.close - new_df.close.shift(1)) / new_df.close.shift(1)) * new_df.volume
    pvt = []
    cum = 0
    for i in new_df['x'][1:]:
        cum += i
        pvt.append(cum)
    if (pvt[-1] > 0) & (pvt[-1] > pvt[-2]):
        return 1
    else:
        return 0


def checkRSI_5(data, close):
    new_df = data.loc[:, ['close']].copy()
    new_row = pd.DataFrame({'close': [float(close)]})
    new_df = new_df.append(new_row, ignore_index=True)
    new_df['close'] = new_df['close'].astype("float")
    rsi = computeRSI(new_df['close'], 14)
    new_df['stoch_rsi'], D = stochastic(rsi, 3, 3, 14)
    if (new_df['stoch_rsi'].values[-1] > 20) & (new_df['stoch_rsi'].values[-2] < 20):
        return 1
    else:
        return 0


def condition_buy(pvt_bool_d, rsi_bool_d, rsi_bool_5, rsi_bool_15, rsi_bool_1H):
    if (pvt_bool_d == 1) & (rsi_bool_d == 1) & (rsi_bool_5 == 1) & (rsi_bool_15 == 1) & (rsi_bool_1H == 1):
        return 1
    else:
        return 0


with open('config.yml') as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)

go_back = 35  # how much data we need to calculate initial indicators
symbol = 'btcusdt'


daily = get_crypto_data("BTC/USDT", "2017-08-17", str(datetime.today() - timedelta(days=1)).split()[0])
daily['x'] = ((daily.close - daily.close.shift(1)) / daily.close.shift(1)) * daily.volume
one = get_binance_bars(symbol.upper(), '1m', datetime.now() - timedelta(minutes=go_back), datetime.now())[:-1]
five = get_binance_bars(symbol.upper(), '5m', datetime.now() - timedelta(minutes=go_back * 5), datetime.now())[:-1]
fifteen = get_binance_bars(symbol.upper(), '15m', datetime.now() - timedelta(minutes=go_back * 15), datetime.now())[:-1]
hour = get_binance_bars(symbol.upper(), '1h', datetime.now() - timedelta(hours=go_back), datetime.now())[:-1]

SOCKET_1 = f'wss://stream.binance.com:9443/ws/{symbol}@kline_1m'
SOCKET_5 = f'wss://stream.binance.com:9443/ws/{symbol}@kline_5m'
SOCKET_15 = f'wss://stream.binance.com:9443/ws/{symbol}@kline_15m'
SOCKET_1H = f'wss://stream.binance.com:9443/ws/{symbol}@kline_1h'
SOCKET_1D = f'wss://stream.binance.com:9443/ws/{symbol}@kline_1d'

closes_1 = np.array(one['close'])  # we need this later to calculate indicators
closes_5 = np.array(five['close'])
closes_15 = np.array(fifteen['close'])
closes_1H = np.array(hour['close'])

cum_vol = float(0)

t1 = threading.Thread(target=wsthread_1, args=(one,))
t5 = threading.Thread(target=wsthread_5, args=(five,))
t15 = threading.Thread(target=wsthread_15, args=(fifteen,))
t_1 = threading.Thread(target=wsthread_1H, args=(hour,))
t_1D = threading.Thread(target=wsthread_1D, args=(daily,))
t1.start()
t5.start()
t15.start()
t_1.start()
t_1D.start()
