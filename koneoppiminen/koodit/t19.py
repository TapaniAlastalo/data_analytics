import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

forecast_time = 12 # kuukautta
seqlength = 12  # syöteverkon historian pituus

df = pd.read_csv('data/AirPassengers.csv', sep=',', decimal='.')

df['Month'] = pd.to_datetime(df['Month'])
df['Time'] = df.index

df['PassengersLag'] = df['Passengers'].shift(1)
df['PassengersDiff'] = df.apply(lambda row:
                           row['Passengers']-row['PassengersLag'], axis=1)
    
for i in range(1, seqlength):
    df['PassengersDiffLag'+str(i)] = df['PassengersDiff'].shift(i)
    
for i in range(1, forecast_time +1):
    df['PassengersDiffFut'+str(i)] = df['PassengersDiff'].shift(-i)
    
df_train = df.iloc[:-2* forecast_time]
df_train.dropna(inplace=True)
df_test = df.iloc[-2* forecast_time:]

# muuttujien valinta ja skaalaus
input_vars = ['PassengersDiff']
for i in range(1, seqlength):
    input_vars.append('PassengersDiffLag'+str(i))

output_vars = []
for i in range(1, forecast_time +1):
    output_vars.append('PassengersDiffFut'+str(i))
    
scaler = preprocessing.StandardScaler()
scalero = preprocessing.StandardScaler()

X = np.array(df_train[input_vars])
X_scaled = scaler.fit_transform(X)
X_scaledLSTM = X_scaled.reshape(X.shape[0], seqlength, 1)
y = np.array(df_train[output_vars])
y_scaled = scalero.fit_transform(y)

X_test = np.array(df_test[input_vars])
X_testscaled = scaler.transform(X_test)
X_testscaledLSTM = X_testscaled.reshape(
    X_test.shape[0], seqlength, 1)

# Trendin mallinnus lineaarisella regressiolla
modelLR = linear_model.LinearRegression()
XLR = df_train['Time'].values
XLR = XLR.reshape(-1,1)
yLR = df_train['Passengers'].values
yLR = yLR.reshape(-1,1)
modelLR.fit(XLR, yLR)
XLR_test = df_test['Time'].values
XLR_test = XLR_test.reshape(-1,1)
df_test['PassengersAvgPred'] = modelLR.predict(XLR_test)

# trendin kulmakerroin
slope = modelLR.coef_

# LTSM verkon muodostus ja opetus
modelLSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(24, input_shape=(seqlength, 1), 
                         return_sequences=False),
    #tf.keras.layers.LSTM(24, return_sequences=False),
    tf.keras.layers.Dense(forecast_time)
    ])

modelLSTM.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae'])

modelLSTM.fit(X_scaledLSTM, y_scaled, epochs=200, batch_size=seqlength)

# Ennusteen (ennusteDiff + trendi) määritys
ennusteDiff = scalero.inverse_transform(
    modelLSTM.predict(X_testscaledLSTM[forecast_time -1].reshape(1,12,1)))
ennuste = np.zeros(13)
ennuste[0] = df_test['Passengers'][df_test.index[forecast_time -1]]

for i in range(1,13):
    for j in range(1,13):
        ennuste[j] = ennuste[j-1]+ennusteDiff[0][j-1]+slope
ennuste = np.array(ennuste[1:])

# ennusteen piirtäminen
df_pred = df_test[-12:]
df_pred['PassengersPred'] = ennuste

plt.plot(df['Month'].values, df['Passengers'].values, color='black', label='Number of passengers')
plt.plot(df_pred['Month'].values, df_pred['PassengersPred'].values, color='red', label='Prediction')

plt.grid()
plt.legend()
plt.show()

print(mean_absolute_error(df_pred['Passengers'].values,
                          df_pred['PassengersPred'].values))




