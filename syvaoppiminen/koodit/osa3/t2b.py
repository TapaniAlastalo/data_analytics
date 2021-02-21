# Tehtävän toteutus

import pandas as pd
# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df1 = pd.read_csv('c:/data/lampotila_2010.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df1 = df1.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature'})
df1 = df1[['date','Temperature']]
data_2010 = df1

df2 = pd.read_csv('c:/data/sahkonkulutus_2010.csv')
df2 = df2.rename(columns={'Alkuaika UTC':'date','Sähkön kulutus Suomessa':'Electricity Consumption'})
df2 = df2[['date','Electricity Consumption']]
data_2010['Electricity Consumption'] = df2['Electricity Consumption']

# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df3 = pd.read_csv('c:/data/lampotila_2011.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df3 = df3.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature'})
df3 = df3[['date','Temperature']]
data_2011 = df3

df4 = pd.read_csv('c:/data/sahkonkulutus_2011.csv')
df4 = df4.rename(columns={'Lopetusaika UTC':'date','Sähkön kulutus Suomessa':'Electricity Consumption'})
df4 = df4[['date','Electricity Consumption']]
data_2011['Electricity Consumption'] = df4['Electricity Consumption']

data = pd.concat([data_2010[['date','Temperature','Electricity Consumption']],data_2011[['date','Temperature','Electricity Consumption']]])
data = data.set_index('date')
data = data.fillna(method='bfill')
print(data.head())
#%%
# skaalataa arvot
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(data.values)

#%%
# Aikasarjaennustamisessa ominaisuudet ovat X tuntia ennen aikaa t ja ennustettavat arvot X tuntia ajan t jälkeen
import numpy as np
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

past_history = 24*7 # Ennustetaan viimeisen seitsemän päivän tietojen avulla
future_target = 24 # Ennustetaan seuraavat 24 tuntia
STEP = 1 # Datasetissä on tallennettu havainnot tunnin välein. 
TRAIN_SPLIT = len(dataset) // 2 # Jaetaan datasetti kahteen, vuosi 2010 koulutusdataksi ja 2011 testidataksi
train_X, train_y = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
test_X, test_y = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
print (f'Single window of past history : {train_X[0].shape}')
print (f'Target to predict : {train_y[0].shape}')

# Tässä tapauksissa ominaisuuksia on vain yksi kappale ja time_steps on 10 eli kymmenen viimeisintä arvoa.
#train_X = train_X.reshape(train_X.shape[0], 7)#train_X.shape[1], 1)
#test_X =  test_X.reshape(test_X.shape[0], 7)#test_X.shape[1], 1)

print(train_X.shape)
print(train_y.shape)

print(test_X.shape)
print(test_y.shape)

#%%
# Luodaan LSTM - neuroverkko ja koulutetaan se koulutusdatalla.
import tensorflow as tf
input_layer = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2])) # muoto on (None, 168, 2)
lstm1 = tf.keras.layers.LSTM(128,return_sequences=True,activation='relu')(input_layer)
lstm2 = tf.keras.layers.LSTM(32,activation='relu')(lstm1)
drop1 = tf.keras.layers.Dropout(0.5)(lstm2)
out = tf.keras.layers.Dense(future_target, activation='linear')(drop1)
model = tf.keras.Model(inputs=input_layer,
                       outputs=out)
print("train")
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse',metrics=['mse','mae'])
model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=10)

#model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
#model.fit(train_X,train_y, epochs=6)
#test_results = model.evaluate(test_X, test_y, verbose=0)
#predictions = model.predict(test_X)
#print(f"Test loss {test_results[0]}")
print("train done")

#%%
# Luodaan ennustukset testidatalle ja piirretään oikeat arvot sekä ennustetut arvot kuvaajaan.
import matplotlib.pyplot as plt
predictions = model.predict(test_X,verbose=0)
nums_preds = []
nums_real = []
index = 0
# Käydään toistorakenteessa läpi ennustukset
for x in range(len(predictions) // len(predictions[0])):
    for num in predictions[index]:
        nums_preds.append(num)
    for num in test_y[index]:
        nums_real.append(num)
    # Siirrytään 24 tuntia eteenpäin, kun sekvenssi oikeista arvoista ja ennustetuista arvoista on lisätty listaan
    index += len(predictions[index]) - 1
fig, ax = plt.subplots(figsize=(17,11))
ax.plot(nums_preds,label='Predictions')
ax.plot(nums_real,label='True')
ax.legend()
plt.show()

#%%
# Tehtävän vastaukset.  Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. 
# Luo tarvittavat osat viivakaavioon fig - muuttujaan matplotlib - kirjastoa käyttäen. Tuloksena pitäisi olla viivakaavio, jossa on piirrettynä oikeat arvot ja neuroverkon ennustukset.
fig