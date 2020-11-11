import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data/Telco.csv', sep=';', decimal='.', encoding='utf-8')
df.fillna(0, inplace=True)

input_variables = df.iloc[:,0:14]
predict_field = "churn"
 
df_train = df.sample(n = 900, replace = False) 
df_test = df.drop(df_train.index)

# train
X = np.array(df_train.iloc[:,1:40])
y = np.array(pd.get_dummies(df_train[predict_field]))

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(X_scaled.shape[1],)),
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(X_scaled, y, epochs=20, batch_size=1)

# test
X = np.array(df_test.iloc[:,1:40])
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

predictedResults = model.predict(X_scaled)
model.summary()

roundedResults = np.round(predictedResults, 3)
df_test['Churn Riski'] = roundedResults[:,1]

results_fields = ['region', 'tenure', 'age', 'marital', 'income', 'employ', 'gender','churn', 'Churn Riski']
df_results = df_test[results_fields].sample(20)
