import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

days_to_forecast = 50

df = pd.read_csv('data/Kysynta.csv', sep=';', encoding='latin_1')
print(df)

df['KysyntäFuture'] = df['Kysyntä'].shift(days_to_forecast)

df_train = df[:250]
df_test = df[250:]

X = np.array(df_train[['Päivä']])
X = X.reshape(-1,1)
y = np.array(df_train['KysyntäFuture'])

# luodaan sequential tyyppinen neuroverkkomalli
model = tf.keras.Sequential([
    # määritellään neuroverkon piilotettu kerros. 10 neuronia (1 input), activation funktio = sigmoid, input kerros (input_shape) = input arvojen lukumäärä
    keras.layers.Dense(20, activation='sigmoid', input_shape=(1,)),
    # 2. piilotettu kerros
    keras.layers.Dense(20, activation='tanh'),
    # 3. piilotettu kerros
    keras.layers.Dense(20, activation='relu'),
    # mallin output kerros. 1 ulostulo (output). Ei aktivointifunktiota # , activation='softmax'
    keras.layers.Dense(1)])

# optimointialgoritmi Adam algoritmi (learning rate=0.001).
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae'])

# epochs = kuinka monta kertaa opetusdata käydään läpi training vaiheessa (painotus), batch_size = kuinka monen data rivin jälkeen painokertoimia päivitetään (oppiminen)
model.fit(X_scaled, y, epochs = 100, batch_size = 10)
ennuste_train = model.predict(X_scaled)
df_train['Ennuste'] = ennuste_train

X_test = np.array(df_test[['Päivä']])
X_test = X_test.reshape(-1,1)
X_testscaled = scaler.transform(X_test)
ennuste_test = model.predict(X_testscaled)
df_test['Ennuste'] = ennuste_test

plt.scatter(df['Päivä'].values, df['Kysyntä'].values, color='black', s=2)
plt.plot((df_train['Päivä'] + days_to_forecast).values, df_train['Ennuste'].values, color='blue')
plt.plot((df_test['Päivä']+ days_to_forecast).values, df_test['Ennuste'].values, color='red')
plt.show()


df_train_validation = df_train.dropna()
df_test_validation = df_test.dropna()
print("Ennusteen keskivirhe opetusdatassa on %.f" %
      mean_absolute_error(df_train_validation['KysyntäFuture'], df_train_validation['Ennuste']))
print("Ennusteen keskivirhe testidatassa on %.f" %
      mean_absolute_error(df_test_validation['KysyntäFuture'], df_test_validation['Ennuste']))


