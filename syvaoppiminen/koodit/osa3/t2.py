# Tehtävän toteutus

import pandas as pd
# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df = pd.read_csv('c:/data/lampotila_2010.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df = df.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
#df = df.set_index('date')
df = df[['date','Temperature']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_temps_2010 = df

df = pd.read_csv('c:/data/sahkonkulutus_2010.csv')
df = df.rename(columns={'Alkuaika UTC':'date','Sähkön kulutus Suomessa':'Electricity Consumption'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
#df = df.set_index('date')
df = df[['date','Electricity Consumption']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_elec_2010 = df

# Yhdistä 2010 datat train dataksi
#df_2010 = pd.concat(df_temps_2010, df_elec_2010, how='right', on=['date'])
#df_2010 = pd.concat([df_temps_2010, df_elec_2010], ignore_index=True)
#df_2010 = pd.concat([df_temps_2010, df_elec_2010], join='inner')
#df_2010 = pd.concat([df_temps_2010, df_elec_2010], axis=1)
df_2010 = pd.concat([df_temps_2010, df_elec_2010], axis=1, ignore_index=True)
#df_2010 = df_temps_2010.append(df_elec_2010, ignore_index=False)
#print(df_2010.head(10))
#df_2010 = df_temps_2010.append(df_elec_2010, ignore_index=False)
#print(df_2010.head(10))


#df_2010 = df_temps_2010
df_2010 = df_2010.rename(columns={0:'Date', 1:'Temperature', 3:'Electricity Consumption'})
#df_2010['Electricity Consumption'] = df_elec_2010[['Electricity Consumption']]
df_2010 = df_2010[['Date', 'Temperature', 'Electricity Consumption']]
df_2010 = df_2010.set_index('Date')
print(df_2010.head())


# Yhdistä 2011 datat test dataksi

# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df = pd.read_csv('c:/data/lampotila_2011.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df = df.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
#print(df)
df['date'] = df['date']
#df = df.set_index('date')
#df = df[['Temperature']]
df = df[['date','Temperature']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_temps_2011 = df

df = pd.read_csv('c:/data/sahkonkulutus_2011.csv')
df = df.rename(columns={'Lopetusaika UTC':'date','Sähkön kulutus Suomessa':'Electricity Consumption'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
#df = df.set_index('date')
#df = df[['Electricity Consumption']]
df = df[['date','Electricity Consumption']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
df_elec_2011 = df

df_2011 = pd.concat([df_temps_2011, df_elec_2011], axis=1, ignore_index=True)
df_2011 = df_2011.rename(columns={0:'Date', 1:'Temperature', 3:'Electricity Consumption'})
#df_2011['Electricity Consumption'] = df_elec_2011[['Electricity Consumption']]
df_2011 = df_2011[['Date', 'Temperature', 'Electricity Consumption']]
df_2011 = df_2011.set_index('Date')
print(df_2011.head())
#df_2011 = df_2011.set_index(1)
#print(df_2011.head())
#%%
# Arvot mitattu joka tunti, jolloin viikossa tunteja 24*7
x_seq = df_2010[0:24*7]['Temperature'].values
# Ennustetaan viikon jälkeisen päivän lämpötilat, eli datasetissä kahdeksas päivä
y_seq = df_2010[24*7:24*8]['Temperature'].values
print(x_seq.shape)
print(y_seq.shape)

#%%
# skaalataa arvot
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dataset_2010 = scaler.fit_transform(df_2010.values)
dataset_2011 = scaler.fit_transform(df_2011.values)

#%%
# History parametrissä määritetään kuinka monella edeltävällä tunnilla ennustetaan ja forecast parametrissä kuinka monta tuntia tulevaisuuteen ennustetaan.
import numpy as np
def dataframe_to_sequences(df,history,forecast):
    x_sequences = []
    y_sequences = []
    i = 0
    while i < (len(df)):
        # Varmistetaan ettei hypätä indeksimuuttujassa Dataframen viimeisen indeksin yli
        if i+history+forecast < len(df):
            print(df)
            x_seq = df[i:i+history].values
            y_seq = df[i+history:i+history+forecast].values
        # lisätään sekvenssit listaan
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
        # Siirretään indeksiä 8 päivää eteenpäin, jolloin voimme luoda seuraavat sekvenssiparit
        i += history + forecast
    return np.array(x_sequences), np.array(y_sequences)
train_X, train_y = dataframe_to_sequences(dataset_2010,24*7,24)
test_X, test_y = dataframe_to_sequences(dataset_2010,24*7,24)
print(train_X.shape)
print(train_y.shape)

print(test_X.shape)
print(test_y.shape)

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
#TRAIN_SPLIT = len(dataset) // 2 # Jaetaan datasetti kahteen, vuosi 2016 koulutusdataksi ja 2017 testidataksi
train_X, train_y = multivariate_data(dataset_2010, dataset_2010[:, 1], 0, None, 
                                                 past_history, future_target, STEP)
test_X, test_y = multivariate_data(dataset_2011, dataset_2011[:, 1], 0, None,
                                                 past_history, future_target, STEP)
print (f'Single window of past history : {train_X[0].shape}')
print (f'Target to predict : {train_y[0].shape}')

# Tässä tapauksissa ominaisuuksia on vain yksi kappale ja time_steps on 10 eli kymmenen viimeisintä arvoa.
train_X = train_X.reshape(train_X.shape[0], 7)#train_X.shape[1], 1)
test_X =  test_X.reshape(test_X.shape[0], 7)#test_X.shape[1], 1)

print(train_X.shape)
print(train_y.shape)

print(test_X.shape)
print(test_y.shape)

#%%
# Luodaan LSTM - neuroverkko ja koulutetaan se koulutusdatalla.
import tensorflow as tf
input_layer = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2], 1)) # muoto on (None, 168, 2)
lstm1 = tf.keras.layers.LSTM(64,return_sequences=True)(input_layer)
lstm2 = tf.keras.layers.LSTM(8,activation='relu')(lstm1)
#lstm3 = tf.keras.layers.Dense(1,activation='linear')(lstm2)
out = tf.keras.layers.Dense(future_target)(lstm2)
model = tf.keras.Model(inputs=input_layer,
                       outputs=out)
print("train")
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse',metrics=['mse','mae'])
model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=6)

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