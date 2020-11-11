import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data/Telco.csv', sep=';', decimal='.', encoding='utf-8')
df.fillna(0, inplace=True)

#input_variables = ['region', 'tenure', 'age', 'marital', 'income', 'employ', 'gender', 'tollfree', 'wireless', 'cardten', 'logtoll', 'logcard', 'custcat']
#input_variables = df.iloc[:,0:40]
#input_variables = df.iloc[:,0:20]
#input_variables = df.iloc[:,0:14]
input_variables = df.iloc[:,0:10]
predict_field = "churn"
 
df_train = df.sample(n = 900, replace = False) 
df_test = df.drop(df_train.index)

#y = np.array(pd.get_dummies(df['fruit_name']))

# train
X = np.array(df_train.iloc[:,1:40])

#y_int = np.array(df_train[predict_field])
#y = to_categorical(y_int)
y = np.array(pd.get_dummies(df_train[predict_field]))

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

model = keras.Sequential([
    # 1. piilotettu / input kerros
    keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(X_scaled.shape[1],)),
    # 2. piilotetu kerros
    keras.layers.Dense(30, activation=tf.nn.relu),
    # output kerros -> 4 output luokkaa, softmax tulostaa ko. luokan todennäköisyyden
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

# hakee sarakkeesta ennusteen, jonka todennäköisyys suurin
predictedResults = model.predict(X_scaled)
ennuste = np.argmax(predictedResults, axis=1)
df_test['Ennuste'] = ennuste
roundedResults = np.round(predictedResults, 3)
df_test['Churn Riski'] = roundedResults[:,1]
#predictedValue = np.max(prediction, axis=1)
#df_test['Churn Riski'] = predictedValue


#results_fields = ['region', 'tenure', 'age', 'marital', 'income', 'employ', 'gender', 'tollfree', 'wireless', 'cardten', 'logtoll', 'logcard', 'custcat','churn', 'Churn Riski']
results_fields = ['region', 'tenure', 'age', 'marital', 'income', 'employ', 'gender','churn', 'Ennuste', 'Churn Riski']
df_results = df_test[results_fields].sample(20)
