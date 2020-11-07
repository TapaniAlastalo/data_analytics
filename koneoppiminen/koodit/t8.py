import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data/fruit_data.csv', sep=',', encoding='utf-8')

X = np.array(df[['mass', 'width', 'height', 'color_score']])

y = np.array(pd.get_dummies(df['fruit_name']))

# Skaalataan X arvot keskiarvoon 0 ja keskihajontaan 1
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)


model = keras.Sequential([
    # 1. piilotettu / input kerros
    keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(X_scaled.shape[1],)),
    # 2. piilotetu kerros
    keras.layers.Dense(30, activation=tf.nn.relu),
    # output kerros -> 4 output luokkaa, softmax tulostaa ko. luokan todennäköisyyden
    keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(X_scaled, y, epochs=20, batch_size=1)

# hakee sarakkeesta ennusteen, jonka todennäköisyys suurin
ennuste = np.argmax(model.predict(X_scaled), axis=1)
df['Ennuste'] = ennuste
