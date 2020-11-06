import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing

days_to_forecast = 30

df = pd.read_csv('data/Google_Stock_Price.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = df.apply(lambda row: len(df) - row.name, axis=1)
df['CloseFuture'] = df['Close'].shift(days_to_forecast)


df_test = df[:185]
df_train = df[185:]

X = np.array(df_train[['Time']])
X = X.reshape(-1,1)
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = np.array(df_train['CloseFuture'])

# luodaan sequential tyyppinen neuroverkkomalli
model = tf.keras.Sequential([
    # määritellään neuroverkon piilotettu kerros. 10 neuronia (1 input), activation funktio = sigmoid, input kerros (input_shape) = input arvojen lukumäärä
    keras.layers.Dense(20, activation='sigmoid', input_shape=(1,)),
    # 2. piilotettu kerros
    keras.layers.Dense(20, activation='sigmoid'),
    # 3. piilotettu kerros
    keras.layers.Dense(20, activation='relu'),
    # mallin output kerros. 1 ulostulo (output). Ei aktivointifunktiota # , activation='softmax'
    keras.layers.Dense(1)])

# optimointialgoritmi Adam algoritmi (learning rate=0.001).
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), #'adam', #tf.train.AdamOptimizer(0.001),
              loss='mse', #'categorical_crossentropy',
              metrics=['mae']) # ['accuracy'])

# epochs = kuinka monta kertaa opetusdata käydään läpi training vaiheessa (painotus), batch_size = kuinka monen data rivin jälkeen painokertoimia päivitetään (oppiminen)
model.fit(X_scaled, y, epochs = 100, batch_size = 10)
ennuste_train = model.predict(X_scaled)
df_train['Ennuste'] = ennuste_train

X_test = np.array(df_test[['Time']])
X_test = X_test.reshape(-1,1)
X_testscaled = scaler.transform(X_test)
ennuste_test = model.predict(X_testscaled)
df_test['Ennuste'] = ennuste_test

plt.scatter(df['Date'].values, df['Close'].values, color='black', s=2)
plt.plot((df_train['Date'] + pd.DateOffset(days=30)).values, df_train['Ennuste'].values, color='blue')
plt.plot((df_test['Date'] + pd.DateOffset(days=30)).values, df_test['Ennuste'].values, color='red')
plt.show()


df_train_validation = df_train.dropna()
df_test_validation = df_test.dropna()
print("Ennusteen keskivirhe opetusdatassa on %.f" %
      mean_absolute_error(df_train_validation['CloseFuture'], df_train_validation['Ennuste']))
print("Ennusteen keskivirhe testidatassa on %.f" %
      mean_absolute_error(df_test_validation['CloseFuture'], df_test_validation['Ennuste']))

# print('Mallin kertoimet ovat \n', model.coef_, model.intercept_)


