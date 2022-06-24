# BACKTESTING

# Importamos librerias
import requests
import pandas as pd
import math
import time
import random
from sklearn.preprocessing import OneHotEncoder
import statistics
import yfinance as yf
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

# Fijamos semilla aleatoria por reproducibilidad
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(66)
random.seed(66)

# Nombres de sectores y ETFs

sectors = ['QQQ', 'XLV', 'VNQ', 'XLY', 'XLF', 'XLI', 'XLE', 'XLU', 'XLP', 'XLB']
sector_names = ['tech', 'healthcare', 'real_state', 'consumer_disc', 'financials',
                'industrials', 'energy', 'utilities', 'consumer_staples', 'materials']


# DESCARGA DE DATOS

## Funcion descarga de datos AlphaVantage

def bajar_datos(time, tick, interval, time_serie, fecha_inicial, fecha_final):
    API_URL = "https://www.alphavantage.co/query"
    symbol = 'SPY'

    data = {
        "function": time,
        "symbol": tick,
        "interval": interval,
        "outputsize": "full",
        "datatype": "json",
        "apikey": "UBU5L7O3UE0ZRFAW"}

    response = requests.get(API_URL, data)
    response_json = response.json()

    data = pd.DataFrame.from_dict(response_json[time_serie], orient='index').sort_index(axis=1)
    data = data.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. adjusted close': 'adjclose',
        '6. volume': 'volume'})
    data = data[['open', 'high', 'low', 'close', 'adjclose', 'volume']]
    data = data.iloc[::-1]
    data = data[fecha_inicial:fecha_final]
    data = data.apply(pd.to_numeric)
    return data


# Funcion indicadores tecnicas

def features_(data, movavg_short, movavg_med, movavg_med2, movavg_slow, std, logret_window, window_volatility):
    data = data.astype(float)
    # Medias moviles
    data['slow_MA'] = data['adjclose'].rolling(window=movavg_short).mean()
    data['med_MA'] = data['adjclose'].rolling(window=movavg_med).mean()
    data['med_MA2'] = data['adjclose'].rolling(window=movavg_med2).mean()
    data['fast_MA'] = data['adjclose'].rolling(window=movavg_slow).mean()

    # Slope medias moviles
    data['fast_slope'] = np.log(data['fast_MA'] / data['fast_MA'].shift())
    data['med_slope'] = np.log(data['med_MA'] / data['med_MA'].shift())
    data['med_slope2'] = np.log(data['med_MA2'] / data['med_MA2'].shift())
    data['slow_slope'] = np.log(data['slow_MA'] / data['slow_MA'].shift())

    # Variaciones logaritmicas
    for dias in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18]:
        data[f'logret_{dias}days'] = np.log(data['adjclose'] / data['adjclose'].shift(dias))

    # Variacion logaritmica ventana para etiquetar
    data['logret_window'] = np.log(data['adjclose'] / data['adjclose'].shift(logret_window))

    # Volatilidad historica
    data['logret'] = np.log(data['adjclose'] / data['adjclose'].shift(1))
    data['volatility'] = data['logret'].rolling(window=window_volatility).std() * np.sqrt(window_volatility)

    data = data.dropna()

    return data


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


# Funcion para etiquetar datos.

def senales_compra_venta(data, umbral_compra, umbral_venta):
    logr_change_window = data['logret_window'].tolist()
    label = []

    umbral_compra_log = np.log(1 + umbral_compra)
    umbral_venta_log = np.log(1 + umbral_venta)

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
    data = data[
        ['fast_slope', 'med_slope', 'med_slope2', 'slow_slope', 'logret_18days', 'logret_15days',
         'logret_12days', 'logret_10days', 'logret_9days', 'logret_8days',
         'logret_7days', 'logret_6days', 'logret_5days', 'logret_4days',
         'logret_3days', 'logret_2days', 'logret', 'volatility', 'label']]
    data_train = data.dropna()
    data = data.iloc[[-1]]
    return data_train, data


print('Descargando datos de AlphaVantage')
raw_data = {}
for etf in sectors:
    x = bajar_datos(time="TIME_SERIES_DAILY_ADJUSTED", tick=etf, interval="daily", time_serie="Time Series (Daily)",
                    fecha_inicial='2004-09-29', fecha_final='2020-12-01')
    raw_data[str(etf)] = x
    # Add volumen maximo para umbral de arrastre
    raw_data[etf]['v_median'] = (raw_data[etf]['volume'].rolling(window=20).median() * 0.01)
    time.sleep(20)
    print(f'Descarga de {etf} realizada')

print('Añadiendo variables técnicas')
features_data = {}
for etf in sectors:
    x = features_(data=raw_data[etf], movavg_short=10, movavg_med=20, movavg_med2=50,
                  movavg_slow=100, std=2, logret_window=20, window_volatility=20)
    x = senales_compra_venta(data=x, umbral_compra=0.03, umbral_venta=0.02)[0]
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

# Balance de clases
balance_sectors = pd.DataFrame()
for etf in sectors:
    label_ = pd.Series(features_data[etf]['label'])
    label_balance = label_.value_counts(normalize=True)
    balance_sectors[etf] = label_balance
#fig, ax = plt.subplots(figsize=(10, 3))
#sns.heatmap(balance_sectors, annot=True, ax=ax)

# CONSTRUCCION DEL MODELO
## Separacion de datos

x = {}
y = {}

ohe = OneHotEncoder()

for etf in sectors:
    x[etf] = features_data[etf].loc[:, features_data[etf].columns != 'label'].values
    y[etf] = features_data[etf].loc[:, features_data[etf].columns == 'label'].values
    y[etf] = ohe.fit_transform(y[etf]).toarray()

X_train = {}
X_test = {}
Y_train = {}
Y_test = {}

for etf in sectors:
    x_train, x_test, y_train, y_test = train_test_split(x[etf], y[etf], test_size=0.3125)
    X_train[etf] = x_train
    X_test[etf] = x_test
    Y_train[etf] = y_train
    Y_test[etf] = y_test

print(X_train['QQQ'].shape)
print(X_test['QQQ'].shape)
print(Y_train['QQQ'].shape)
print(Y_test['QQQ'].shape)

# Importamos librerias Keras y sklearn

import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn import metrics

METRICS = ['accuracy',
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall')]


# Definimos el modelo

def create_model(neurons_1, neurons_2, dropout, optimizer):
    model = Sequential()
    model.add(Dense(neurons_1, kernel_initializer=keras.initializers.glorot_uniform(seed=66),
              input_dim=X_train[etf].shape[1], activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(neurons_2, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=METRICS)
    return model


# Descargamos los mejores hiperparametros de cada sector

with open("best_params_dense.txt", "rb") as myFile:
    best_params = pickle.load(myFile)


# Entrenamiento de los modelos

predictions = {}
acc = {}
loss = {}
precision = {}
recall = {}
all_models = {}

for etf in sectors:
    print(etf)
    print('------------------------------------------------------------------------')

    all_models[etf] = create_model(neurons_1=best_params[etf]['neurons_1'], neurons_2=best_params[etf]['neurons_2'],
                                   dropout=best_params[etf]['dropout'], optimizer=best_params[etf]['optimizer'])

    es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
    # Fit model
    history = all_models[etf].fit(X_train[etf], Y_train[etf], validation_split=0.2, epochs=1000, batch_size=128,
                                  callbacks=[es])

    #plt.plot(history.history["loss"])
    #plt.grid()
    #plt.legend(["loss"])

    # Hacemos prediccion sobre los datos de test
    predictions[etf] = all_models[etf].predict(X_test[etf])

# Métricas de evaluacion y conversion a etiquetas 0, 1, 2

pred = {}
for etf in sectors:
    pred_ = []
    for i in range(0, len(predictions[etf])):
        x = np.argmax(predictions[etf][i] > 0.7)
        pred_.append(x)
    pred[etf] = pred_
    print(etf)
    print(metrics.classification_report(features_data[etf]['label'][-len(predictions[etf]):], pred[etf]))


# GESTION DE CARTERAS Y TESORERIA

capital_inversion_etf = 10000
capital_inicial = 100000
stop_porc = 0.02

# Formato fecha
for etf in sectors:
    features_data[etf].index = pd.to_datetime(features_data[etf].index, format='%Y-%m-%d').strftime('%Y-%m-%d')

signals = {}
fecha_inicial = features_data['QQQ'].index[-len(predictions[etf])]
fecha_final = features_data['QQQ'].index[-1]

for etf in sectors:
    signals[etf] = pd.DataFrame()
    signals[etf]['adjclose'] = raw_data[etf]['adjclose']
    signals[etf] = signals[etf][fecha_inicial:fecha_final]
    # Añadimos predicciones de compras y ventas
    signals[etf]['pred'] = pred[etf]


## Control de capital disponible, beneficos, acciones, flujos de caja

# Ejecucion de las ordenes al dia siguiente
for etf in sectors:
    signals[etf]['pred'] = signals[etf]['pred'].shift(1)
    signals[etf] = signals[etf].dropna()

fecha_inicial_ = signals['QQQ'].index[0]
fecha_final_ = signals['QQQ'].index[-1]

tesoreria_etf = {}

# Backtesting
for sector in sectors:
    capital_inversion = 10000

    df = signals[sector]
    df = df.fillna(0)

    df['capital'] = 0
    df['bfos'] = 0
    df['bfos_acumulados'] = 0
    df['dinero_estrategia'] = capital_inversion
    df['volumen_max'] = raw_data[sector]['v_median'][fecha_inicial_:fecha_final_]
    df['stoploss'] = 0

    comision_compra = 5
    comision_venta = 5
    quantity = 0
    bfos_acumulados = 0
    buy_price = 0
    precio_compra=0
    precio_venta=0

    comprado = False
    for index, row in df.iterrows():

        accion = row['pred']
        price = row['adjclose']
        volumen = row['volumen_max']
        # Compra
        if (accion == 1) & (not comprado):
            # Quito las comisiones para asegurarme de poder pagarlas
            quantity = int((capital_inversion - (comision_compra + comision_venta)) / price)
            # Compruebo si puede comprar
            if (quantity == 0) | (capital_inversion <= comision_compra + price):
                print(f"No puedo comprar. Fondos insuficientes. Balance: {capital_inversion}. Precio: {price}")
                continue

            if (volumen > capital_inversion):
                capital_inversion -= quantity * price + comision_compra
                buy_price = price
                comprado = True
                df.loc[index, 'precio_compra'] = buy_price
                df.loc[index, 'n_acciones'] = quantity
            elif (volumen <= capital_inversion):
                continue


        # Venta
        if (accion == 2) & comprado:
            capital_inversion += quantity * price - comision_venta
            bfos = (price - buy_price) * quantity
            bfos_acumulados += bfos - comision_venta
            df.loc[index, 'bfos'] = round(bfos, 2)
            comprado = False
            df.loc[index, 'precio_venta'] = precio_venta + price

        # Venta stoploss
        elif (price < (buy_price*(1-stop_porc))) & comprado:
            capital_inversion += quantity * price - comision_venta
            bfos = (price - buy_price) * quantity
            bfos_acumulados += bfos - comision_venta
            df.loc[index, 'bfos'] = round(bfos, 2)
            comprado = False
            df.loc[index, 'precio_venta'] = price
            df.loc[index, 'stoploss'] = 1

        df.loc[index, 'capital'] = round(capital_inversion, 2)
        df.loc[index, 'bfos_acumulados'] = round(bfos_acumulados, 2)

    df['dinero_estrategia'] = df['dinero_estrategia'] + df['bfos_acumulados']
    tesoreria_etf[sector] = df
    print(f"Backtesting {sector} realizado")

# RESULTADOS

precio_compra = {}
precio_venta = {}
n_acciones = {}

for etf in sectors:
    precio_compra[etf] = [i for i in tesoreria_etf[etf]['precio_compra'] if str(i) != 'nan']
    precio_venta[etf] = [i for i in tesoreria_etf[etf]['precio_venta'] if str(i) != 'nan']
    n_acciones[etf] = [i for i in tesoreria_etf[etf]['n_acciones'] if str(i) != 'nan']

lista_n_operaciones = []
lista_beneficio_accion = []
lista_operaciones_beneficio = []
resultados = {}
for etf in sectors:
    resultados[etf] = pd.DataFrame()
    resultados[etf]['precio_compra'] = precio_compra[etf]
    resultados[etf].loc[:, 'precio_venta'] = pd.Series(precio_venta[etf])
    resultados[etf]['n_acciones'] = n_acciones[etf]
    resultados[etf]['cantidad_compra'] = resultados[etf]['precio_compra'] * resultados[etf]['n_acciones']
    resultados[etf]['cantidad_venta'] = resultados[etf]['precio_venta'] * resultados[etf]['n_acciones']
    resultados[etf]['beneficio'] = resultados[etf]['cantidad_venta'] - resultados[etf]['cantidad_compra'] - (comision_compra + comision_venta)
    resultados[etf]['operaciones_beneficio'] = np.where(resultados[etf]['beneficio'] > 0, 1, 0)

    # Beneficio medio por sector
    beneficio_accion = resultados[etf]['beneficio'].mean()
    lista_beneficio_accion.append(beneficio_accion)
    # % operaciones positivas por sector
    op_pos = resultados[etf]['operaciones_beneficio'].sum()
    op_pos = op_pos / (resultados[etf].shape[0]) * 100
    lista_operaciones_beneficio.append(op_pos)
    # Numero operaciones
    op = resultados[etf].shape[0]
    lista_n_operaciones.append(op)


# Beneficio medio por sector
profit_mean = statistics.mean(lista_beneficio_accion)
# Media del % de operaciones con beneficios
positive_op_mean = statistics.mean(lista_operaciones_beneficio)
# Media numero de operaciones
operaciones_mean = statistics.mean(lista_n_operaciones)

# Graficos
def plot_bars(data, y_label, x_label, mean, labels):
    frequencies = pd.DataFrame(data)
    plt.figure()
    ax = frequencies.plot(kind='bar', color='cornflowerblue')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(labels, rotation='horizontal')
    ax.legend().remove()
    hline = plt.axhline(y=mean, color='r')
    plt.axhline(y=0, color='black')
    rects = ax.patches

    for rect, benef in zip(rects, data):
        height = rect.get_height()
        if height > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, height + (height * 0.02), "%.0f" % benef,
                    ha='center', va='bottom', fontsize=12)
        else:
            ax.text(rect.get_x() + rect.get_width() / 2, height - 8, "%.0f" % benef,
                    ha='center', va='bottom', fontsize=12)
    if (min(data) <= 0) & (max(data) >= 0):
        plt.ylim([min(data) + (min(data) * 0.5), max(data) + (max(data) * 0.4)])
        plt.legend([hline], [f'Mean = {"%.3f" % mean}'])
    elif (max(data) == 0):
        plt.ylim([min(data) + (min(data) * 0.2), -(min(data) * 0.2)])
        ax.legend().remove()
    else:
        plt.ylim([0, max(data) + (max(data) * 0.4)])
        plt.legend([hline], [f'Mean = {"%.2f" % mean}'])
    #plt.savefig('beneficio.png', dpi=500)

plot_bars(lista_beneficio_accion, 'Beneficio por operacion', 'Sectors', profit_mean, sectors)
plot_bars(lista_n_operaciones, 'Numero de operaciones', 'Sectors', operaciones_mean, sectors)
plot_bars(lista_operaciones_beneficio, '% operaciones con beneficio', 'Sectors', 50, sectors)

# Rentabilidad por sector vs estrategia Buy and Hold sector

# Buy and Hold por sector
buy_hold_etf = pd.DataFrame()
for etf in sectors:
    buy_hold_etf[etf] = (np.log(
        signals[etf]['adjclose'] / signals[etf]['adjclose'].shift(1))) * capital_inversion_etf

color_graph = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
for etf, name, color in zip(sectors, sector_names, color_graph):
    plt.subplots(figsize=(10, 5))
    plt.title(f'Rentabilidad por sector: {name} ({etf})', fontsize=20)
    rentabilidad_sector = tesoreria_etf[etf]['dinero_estrategia']
    buy_hold = capital_inversion_etf + buy_hold_etf[etf].cumsum()
    plt.legend(sectors)
    rentabilidad_sector.plot(color=color)
    buy_hold.plot(color='blue', alpha=0.3)

# Rentabilidad estrategia completa (todos los sectores)

dinero_estrategia = pd.DataFrame()
for etf in sectors:
    dinero_estrategia[etf] = tesoreria_etf[etf]['dinero_estrategia']
dinero_estrategia['sum_capital'] = dinero_estrategia.sum(axis=1)

# Construccion del indice sintetico

ind_sintetico = pd.DataFrame()
for etf in sectors:
    ind_sintetico[etf] = tesoreria_etf[etf]['adjclose']

ind_composicion = ind_sintetico.copy() > 0.001
ind_composicion *= 1
ind_composicion['sum_row'] = ind_composicion.sum(axis=1)
ind_composicion = ind_composicion[sectors].div(ind_composicion.sum_row, axis=0)

ind_invertido = pd.DataFrame(ind_composicion * capital_inicial)
pd.DataFrame.reset_index(ind_invertido, drop=True, inplace=True)

ind_rentabilidad = pd.DataFrame()
for etf in sectors:
    ind_rentabilidad[etf] = np.diff(np.log(ind_sintetico[etf]))

ind_resultado = ind_rentabilidad * ind_invertido
ind_resultado['rent'] = ind_resultado.sum(axis=1)
ind_resultado['evolucion_inv'] = capital_inicial + ind_resultado['rent'].cumsum()

## Comparacion estrategia ETF vs indice sintetico
evolucion_estrategia = dinero_estrategia['sum_capital']
evolucion_indice = ind_resultado['evolucion_inv']
plt.subplots(figsize=(15, 6))
plt.title('Indice sintetico vs Estrategia ETF', fontsize=20)
evolucion_indice.plot(color='blue', alpha=0.3)
evolucion_estrategia.plot(color='orange')
plt.gca().legend(('Indice sintetico', 'Estrategia'))

dinero_estrategia.index = pd.to_datetime(dinero_estrategia.index, format='%Y-%m-%d')
ind_resultado.index = dinero_estrategia.index

# Rentabilidad y alpha anual de la estrategia
alpha_cartera = pd.DataFrame()
alpha_cartera['capital_estrategia_anual'] = dinero_estrategia.resample('Y').ffill()['sum_capital']
alpha_cartera['capital_ind_anual'] = ind_resultado.resample('Y').ffill()['evolucion_inv']
alpha_cartera['year'] = range(0, alpha_cartera.shape[0])
inicial_year_ind = alpha_cartera['capital_ind_anual'][0]
inicial_year_strat = alpha_cartera['capital_estrategia_anual'][0]
alpha_cartera['rent_anual_ind'] = (alpha_cartera['capital_ind_anual'] / inicial_year_ind) ** (
            1 / alpha_cartera['year']) - 1
alpha_cartera['rent_anual_strat'] = (alpha_cartera['capital_estrategia_anual'] / inicial_year_strat) ** (
            1 / alpha_cartera['year']) - 1
alpha_cartera['alpha'] = alpha_cartera['rent_anual_strat'] - alpha_cartera['rent_anual_ind']

# Rentabilidad anual estrategia year 2020
rent_anual_cartera = round(alpha_cartera['rent_anual_strat'][-1] * 100, 3)
print(f'Rentabilidad anual de la cartera: {rent_anual_cartera} %')

# Rentabilidad indice sintetico year 2020
rent_anual_indice = round(alpha_cartera['rent_anual_ind'][-1] * 100, 3)
print(f'Rentabilidad anual del indice: {rent_anual_indice} %')

# Grafico alpha anual
year_a = []
for i in alpha_cartera.index:
    x = str(i).split('-')[0]
    year_a.append(x)

frequencies = pd.DataFrame(alpha_cartera['alpha'])
alpha_ = alpha_cartera['alpha']
plt.figure()
ax = frequencies.plot(kind='bar', color='cornflowerblue')
ax.set_xlabel('Years')
ax.set_ylabel('Alpha')
ax.set_xticklabels(year_a, rotation='horizontal')
ax.legend().remove()
hline = plt.axhline(y=alpha_cartera['alpha'].mean(), color='r')
plt.axhline(y=0, color='black')
plt.ylim([min(alpha_) + (min(alpha_) * 0.1), 0 - (min(alpha_) * 0.2)])
plt.legend([hline], [f'Mean = {"%.2f" % alpha_cartera["alpha"].mean()}'], loc='lower right')
rects = ax.patches
for rect, benef in zip(rects, alpha_):
    height = rect.get_height()
    if height >= 0:
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.005, "%.0f" % benef,
                ha='center', va='bottom', fontsize=12)
    else:
        ax.text(rect.get_x() + rect.get_width() / 2, height - 0.0115, "%.2f" % benef,
                ha='center', va='bottom', fontsize=12)

# Graficos de compras y ventas por sector
data_graf = {}
for etf in sectors:
    data_graf[etf] = pd.DataFrame()
    data_graf[etf]['close'] = tesoreria_etf[etf]['adjclose']
    data_graf[etf]['buy_signal'] = tesoreria_etf[etf]['precio_compra'].replace(0, np.nan)
    data_graf[etf]['sell_signal'] = tesoreria_etf[etf]['precio_venta'].replace(0, np.nan)
    data_graf[etf] = data_graf[etf][1000:]
    data_graf[etf].index = pd.to_datetime(data_graf[etf].index, format='%Y-%m-%d')

for name, etf, color in zip(sector_names, sectors, color_graph):
    plt.figure(figsize=(15, 7))
    plt.scatter(data_graf[etf].index, data_graf[etf]['buy_signal'], color='green', label='buy', marker='^', alpha=1)
    plt.scatter(data_graf[etf].index, data_graf[etf]['sell_signal'], color='red', label='sell', marker='v', alpha=1)
    plt.title('{}: {} buy and sell signals'.format(name, etf))
    plt.xlabel('date')
    plt.ylabel('close price')
    plt.xticks(rotation=45)
    close = data_graf[etf]['close']
    close.plot(color=[color], alpha=0.3)
    plt.legend()



# Drawdown de la estrategia
window = 252
roll_Max = dinero_estrategia['sum_capital'].rolling(window, min_periods=1).max()
daily_drawdown = dinero_estrategia['sum_capital'] / roll_Max - 1.0
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
daily_drawdown.plot()
max_daily_drawdown.plot()

