# Busqueda de mejores parametros

import requests
import pandas as pd
import numpy as np
import math
import time
import pickle
from sklearn.preprocessing import OneHotEncoder
import yfinance as yf
from sklearn.model_selection import train_test_split


sectors = ['QQQ', 'XLV', 'VNQ','XLY', 'XLF', 'XLI', 'XLE', 'XLU', 'XLP', 'XLB']

def bajar_datos(time, tick, interval, time_serie, fecha_inicial):
    API_URL = "https://www.alphavantage.co/query" 
    symbol = 'SPY'

    data = { "function": time, 
    "symbol": tick,
    "interval": interval,
    "outputsize" : "full",
    "datatype": "json", 
    "apikey": "UBU5L7O3UE0ZRFAW" } 

    response = requests.get(API_URL, data) 
    response_json = response.json()

    data = pd.DataFrame.from_dict(response_json[time_serie], orient= 'index').sort_index(axis=1)
    data = data.rename(columns={ '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adjclose', '6. volume': 'volume'})
    data = data[[ 'open', 'high', 'low', 'close', 'adjclose', 'volume']]
    data = data.iloc[::-1]
    data = data[fecha_inicial:]
    data = data.apply(pd.to_numeric)
    return (data)

# Funcion caracteristicas

def features_(data, movavg_short, movavg_med, movavg_med2, movavg_slow, std, logret_window, window_volatility):
    data = data.astype(float)
      # Medias moviles 
    data['slow_MA'] = data['adjclose'].rolling(window=movavg_short).mean()
    data['med_MA'] = data['adjclose'].rolling(window=movavg_med).mean()
    data['med_MA2'] = data['adjclose'].rolling(window=movavg_med2).mean()
    data['fast_MA'] = data['adjclose'].rolling(window=movavg_slow).mean()

      # Slope
    data['fast_slope'] = np.log(data['fast_MA'] / data['fast_MA'].shift())
    data['med_slope'] = np.log(data['med_MA'] / data['med_MA'].shift())
    data['med_slope2'] = np.log(data['med_MA2'] / data['med_MA2'].shift())
    data['slow_slope'] = np.log(data['slow_MA'] / data['slow_MA'].shift())
      # Porcentaje de cambios en 2, 5, 7 dias

    for dias in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]:
        data[f'logret_{dias}days'] = np.log(data['adjclose'] / data['adjclose'].shift(dias))

      # Porcentaje de cambio en x dias
    data['logret_window'] = np.log(data['adjclose'] / data['adjclose'].shift(logret_window))

      # Volatilidad
    data['logret'] = np.log(data['adjclose'] / data['adjclose'].shift(1))          
    data['volatility'] = data['logret'].rolling(window=window_volatility).std() * np.sqrt(window_volatility) 

    data = data.dropna()

    return (data)

# Descarga caracteristicas por sectores en Yahoo Finance

# QQQ: semiconductores
semic = yf.download("^SOX")
semic['logret_c'] = np.log(semic['Adj Close'] / semic['Adj Close'].shift(1))
semic['logret_5day_c'] = np.log(semic['Adj Close'] / semic['Adj Close'].shift(5))
semic['logret_10day_c'] = np.log(semic['Adj Close'] / semic['Adj Close'].shift(10))

# VNQ: madera
lum = yf.download('LBS=F')
lum['logret_c'] = np.log(lum['Adj Close'] / lum['Adj Close'].shift(1))
lum['logret_5day_c'] = np.log(lum['Adj Close'] / lum['Adj Close'].shift(5))
lum['logret_10day_c'] = np.log(lum['Adj Close'] / lum['Adj Close'].shift(10))

# XLB: fabricante maquinaria
cat = yf.download("CAT")
cat['logret_c'] = np.log(cat['Adj Close'] / cat['Adj Close'].shift(1))
cat['logret_5day_c'] = np.log(cat['Adj Close'] / cat['Adj Close'].shift(5))
cat['logret_10day_c'] = np.log(cat['Adj Close'] / cat['Adj Close'].shift(10))

# XLP: azucar
sug = yf.download("SB=F")
sug['logret_c'] = np.log(sug['Adj Close'] / sug['Adj Close'].shift(1))
sug['logret_5day_c'] = np.log(sug['Adj Close'] / sug['Adj Close'].shift(5))
sug['logret_10day_c'] = np.log(sug['Adj Close'] / sug['Adj Close'].shift(10))

# XLV: Service Corporation International
health = yf.download("SCI")
health['logret_c'] = np.log(health['Adj Close'] / health['Adj Close'].shift(1))
health['logret_5day_c'] = np.log(health['Adj Close'] / health['Adj Close'].shift(5))
health['logret_10day_c'] = np.log(health['Adj Close'] / health['Adj Close'].shift(10))

# XLF y XLU: bonos a 1 year
bond = yf.download("^TNX")
bond['logret_c'] = np.log(bond['Adj Close'] / bond['Adj Close'].shift(1))
bond['logret_5day_c'] = np.log(bond['Adj Close'] / bond['Adj Close'].shift(5))
bond['logret_10day_c'] = np.log(bond['Adj Close'] / bond['Adj Close'].shift(10))

# XLE: petroleo
oil = yf.download("CL=F")
# un dia el petroleo tuvo precio negativo
oil['Adj Close'] = np.where(oil['Adj Close'] < 0, 0.01, oil['Adj Close'])
oil['logret_c'] = np.log(oil['Adj Close'] / oil['Adj Close'].shift(1))
oil['logret_5day_c'] = np.log(oil['Adj Close'] / oil['Adj Close'].shift(5))
oil['logret_10day_c'] = np.log(oil['Adj Close'] / oil['Adj Close'].shift(10))

# XLI: United State Steel Corporation
indust = yf.download("X")
indust['logret_c'] = np.log(indust['Adj Close'] / indust['Adj Close'].shift(1))
indust['logret_5day_c'] = np.log(indust['Adj Close'] / indust['Adj Close'].shift(5))
indust['logret_10day_c'] = np.log(indust['Adj Close'] / indust['Adj Close'].shift(10))

# XLY
trans = yf.download("^DJT")
trans['logret_c'] = np.log(trans['Adj Close'] / trans['Adj Close'].shift(1))
trans['logret_5day_c'] = np.log(trans['Adj Close'] / trans['Adj Close'].shift(5))
trans['logret_10day_c'] = np.log(trans['Adj Close'] / trans['Adj Close'].shift(10))


# Funcion para añadir labels. 1: compra 2: vende 0: no hace nada

def señales_compra_venta(data, umbral_compra, umbral_venta):
    logr_change_window = data['logret_window'].tolist()
    label = []

    umbral_compra_log = np.log(1+umbral_compra)
    umbral_venta_log = np.log(1+umbral_venta)

    for ind, i in enumerate(logr_change_window):
        if i > umbral_compra_log: 
            label.append(1)
        elif i < -umbral_venta_log:
              label.append(2)
        elif i > -umbral_venta_log or i < umbral_compra_log:
              label.append(0)
        elif math.isnan(i) == True:
              label.append(float('nan'))

    data['label'] = label
    data['label'] = data['label'].shift(-1)
    data = data[['fast_slope', 'med_slope', 'med_slope2','slow_slope', 'logret_15days', 'logret_12days', 
                 'logret_10days', 'logret_9days', 'logret_8days',
                 'logret_7days', 'logret_6days', 'logret_5days','logret_4days',
                 'logret_3days', 'logret_2days','logret', 'volatility', 'label']] 
    data_train = data.dropna()
    data = data.iloc[[-1]]
    return (data_train, data)

print('Descargando datos Alpha Vantage')
raw_data = {}
for etf in sectors:
    x = bajar_datos(time="TIME_SERIES_DAILY_ADJUSTED", tick=etf, interval="daily", time_serie="Time Series (Daily)", fecha_inicial='2004-09-29')
    raw_data[str(etf)] = x
    time.sleep(20)

print('Anadiendo caracteristicas')
features_data = {} 
for etf in sectors:
  x = features_(data=raw_data[etf], movavg_short=5, movavg_med=10, movavg_med2=20, 
                movavg_slow=50, std=2, logret_window=20, window_volatility=20)
  x = señales_compra_venta(data=x, umbral_compra=0.03, umbral_venta=0.02)[0]
  features_data[str(etf)] = x

# Añadimos caractisticas de cada sector

def caracteristicas_sector(features=features_data):
    features['XLB'] = features['XLB'].join(cat['logret_c'])
    features['XLB'] = features['XLB'].join(cat['logret_5day_c'])
    features['XLB'] = features['XLB'].join(cat['logret_10day_c'])
    features['XLY'] = features['XLY'].join(trans['logret_c'])
    features['XLY'] = features['XLY'].join(trans['logret_5day_c'])
    features['XLY'] = features['XLY'].join(trans['logret_10day_c'])
    features['XLP'] = features['XLP'].join(sug['logret_c'])
    features['XLP'] = features['XLP'].join(sug['logret_5day_c'])
    features['XLP'] = features['XLP'].join(sug['logret_10day_c'])
    features['XLF'] = features['XLF'].join(bond['logret_c'])
    features['XLF'] = features['XLF'].join(bond['logret_5day_c'])
    features['XLF'] = features['XLF'].join(bond['logret_10day_c'])
    features['XLE'] = features['XLE'].join(oil['logret_c'])
    features['XLE'] = features['XLE'].join(oil['logret_5day_c'])
    features['XLE'] = features['XLE'].join(oil['logret_10day_c'])
    features['QQQ'] = features['QQQ'].join(semic['logret_c'])
    features['QQQ'] = features['QQQ'].join(semic['logret_5day_c'])
    features['QQQ'] = features['QQQ'].join(semic['logret_10day_c'])
    features['XLU'] = features['XLU'].join(bond['logret_c'])
    features['XLU'] = features['XLU'].join(bond['logret_5day_c'])
    features['XLU'] = features['XLU'].join(bond['logret_10day_c'])
    features['VNQ'] = features['VNQ'].join(lum['logret_c'])
    features['VNQ'] = features['VNQ'].join(lum['logret_5day_c'])
    features['VNQ'] = features['VNQ'].join(lum['logret_10day_c'])
    features['XLI'] = features['XLI'].join(indust['logret_c'])
    features['XLI'] = features['XLI'].join(indust['logret_5day_c'])
    features['XLI'] = features['XLI'].join(indust['logret_10day_c'])
    features['XLV'] = features['XLV'].join(health['logret_c'])
    features['XLV'] = features['XLV'].join(health['logret_5day_c'])
    features['XLV'] = features['XLV'].join(health['logret_10day_c'])


caracteristicas_sector(features=features_data)

for etf in sectors:
  features_data[etf] = features_data[etf].fillna(0)

x = {}
y = {}

ohe = OneHotEncoder()

for etf in features_data:
    x[str(etf)] = features_data[etf].loc[:, features_data[etf].columns != 'label'].values
    y[etf] = features_data[etf].loc[:, features_data[etf].columns == 'label'].values
    y[etf] = ohe.fit_transform(y[etf]).toarray()

X_train = {}
X_test = {}
Y_train = {}
Y_test = {}

for etf in sectors:
  x_train, x_test, y_train, y_test = train_test_split(x[etf], y[etf], test_size=0.3125)
  X_train[etf] = x_train
  X_test[etf]  = x_test
  Y_train[etf] = y_train
  Y_test[etf] = y_test

from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(neurons_1, neurons_2, dropout, optimizer):
  model = Sequential()
  model.add(Dense(neurons_1,input_dim=X_train[etf].shape[1], activation='relu'))
  model.add(Dropout(dropout))
  model.add(Dense(neurons_2,activation='relu'))
  model.add(Dense(256, activation = 'relu'))
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(3,activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
  return model

best_params = {}
for etf in sectors:  
  # create model
  model = KerasClassifier(build_fn=create_model, batch_size=128, epochs=50,verbose=0)
  # define the grid search parameters
  neurons_1 = [32, 64, 128]
  neurons_2 = [96, 128, 256]
  dropout = [0.1, 0.3, 0.5]
  optimizer = ['adam', 'nadam']
  param_grid = dict(neurons_1 = neurons_1, neurons_2 = neurons_2, dropout = dropout, optimizer=optimizer)
  grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=4)
  grid_result = grid.fit(X_train[etf], Y_train[etf])
  # summarize results
  best_params[etf]=grid_result.best_params_
  print(etf)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))
  print('------------------------------------------------------------------------')

# Para guardar los mejores parametros

#with open("best_params_dense.txt", "wb") as myFile:
    #pickle.dump(best_params, myFile)

#with open("best_params_dense.txt", "rb") as myFile:
    #best_params = pickle.load(myFile)

