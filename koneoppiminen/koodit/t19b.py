import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data/MachineData.csv', sep=';', decimal='.', encoding='utf-8')
 
teamId = {'TeamA':1,  'TeamB':2, 'TeamC':3}
df['TeamId'] = df['Team'].map(teamId)
df['TeamId'].fillna(-1, inplace=True)

providerId = {'Provider1':1, 'Provider2':2, 'Provider3':3, 'Provider4':4}
df['ProviderId'] = df['Provider'].map(providerId)
df['ProviderId'].fillna(-1, inplace=True)

df_train = df.sample(n = 200, replace = False) 
df_test = df.drop(df_train.index)

input_variables = ['TeamId', 'ProviderId', 'Lifetime', 'PressureInd', 'MoistureInd', 'TemperatureInd']
X = np.array(df[input_variables])

y = np.array(pd.get_dummies(df['Broken']))
#y = np.array((df['Broken']))

# Skaalataan X arvot keskiarvoon 0 ja keskihajontaan 1
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)


model = keras.Sequential([
    # 1. piilotettu / input kerros
    keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(X_scaled.shape[1],)),
    # 2. piilotetu kerros
    keras.layers.Dense(30, activation=tf.nn.relu),
    # output kerros -> 2 output luokkaa, softmax tulostaa ko. luokan todennäköisyyden
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(X_scaled, y, epochs=20, batch_size=1)

predictedResults = model.predict(X_scaled)
model.summary()
roundedResults = np.round(predictedResults, 3)
df['Breakdown Risk'] = roundedResults[:,1]
#ennuste = np.argmax(predictedResults, axis=1)
#df['Prediction'] = ennuste

dfResults1 = df.iloc[:, [0,1,2,3,4,5,6,7,10]].sample(20)

dfResults2 = df[df['Broken']==0]
dfResults2.sort_values(by=['Breakdown Risk'], ascending=False, inplace=True)

rikkoutuvat = dfResults2.iloc[0:10, [0,10]]