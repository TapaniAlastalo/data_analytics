# Aikasarjaennustaminen

# Aikasarjaennustaminen on historiatietojen pohjalta tulevaisuuden ennustamista.
# Näitä käyttötapauksia olisivat esim. lämpötilan ennustaminen mittaushistorian perusteella
# tai osakekurssin tulevien hintojen mallintaminen osakkeen hintahistorian perusteella.

# Yleensä aikasarjaennustamisen ongelmien ratkaisu ei onnistu pelkästään ennustettavan arvon
# historiatiedoista ennustamalla.
# Lämpötilan historiatietojen perusteella ei saa luotettavaa ennustetta usean kymmenen vuoden päähän,
# johtuen mm. ilmastonmuutoksesta.
# Myös yrityksen osakekurssin aikaisempi kasvu ei takaa, että yrityksen osakekurssi kasvaisi loputtomiin.

# Mitä enemmän muuttujia, jonka avulla ennustetaan, sitä parempia ennustettavia malleja voidaan rakentaa.
# Aikaisemmassa RNN - materiaaleissa ennustettiin sinifunktion seuraavaa arvoa,
# mutta oikeassa maailmassa ennustusaikaväli tulisi olla pidempi,
# eikä vain seuraavan arvon ennustamista tulevaisuuteen.
# Lämpötilaa ennustattessa haluttaisiin 24 tunnin ennustuksia tai viikon ennustuksia.

# Ladataan Ilmatieteenlaitoksen avoimesta rajapinnasta vuoden 2016 ja 2017 lämpötilat ja ilmanpaineet,
# jotka on mitattu joka tunti.
# Luodaan neuroverkkomalli, joka ennustaisi vuoden 2016 lämpötilojen ja ilmanpaineen avulla vuoden 2017 lämpötilat.

import pandas as pd
# Käytetään parse_dates parametriä luomaan yksi aikaleima sarake "Vuosi", "Kk", "Pv" ja "Klo" sarakkeista
df = pd.read_csv('2016-2017-lampotila.csv',parse_dates=[['Vuosi','Kk','Pv','Klo']])
df = df.rename(columns={'Vuosi_Kk_Pv_Klo':'date','Ilman lämpötila (degC)':'Temperature','Ilmanpaine (msl) (hPa)':'Air Pressure'})
# Asetetaan aikaleima DataFramen indeksiksi. Voimme kätevästi pilkkoa esim. päiviä DataFramesta indeksin avulla.
df = df.set_index('date')
df = df[['Temperature','Air Pressure']]
# Täytetään NaN arvot ajallisesti seuraavalla arvolla.
df = df.fillna(method='bfill')
print(df.head(10))

'''
                     Temperature  Air Pressure
date                                          
2016-01-01 00:00:00         -4.5        1037.3
2016-01-01 01:00:00         -4.5        1037.5
2016-01-01 02:00:00         -4.5        1037.6
2016-01-01 03:00:00         -4.6        1037.8
2016-01-01 04:00:00         -4.5        1038.0
2016-01-01 05:00:00         -5.1        1038.2
2016-01-01 06:00:00         -5.9        1038.9
2016-01-01 07:00:00         -6.4        1039.6
2016-01-01 08:00:00         -6.4        1040.0
2016-01-01 09:00:00         -6.6        1040.4
'''

#%%

# Sekvenssien luonti
# Datasetti on ladattu nyt Pandasin DataFrameen.
# Luodaan funktio, jonka avulla datasetti jaetaan ominaisuuksiin ja ennustettaviin arvoihin.
# Aikasarjaennustamisessa tämä tarkoittaa sitä,
# että ominaisuuksina on ilman lämpötilan ja ilmanpaineen historiatiedot
# ja ennustettavina arvoina on tuleva lämpötila. Ominaisuuksista tulee siis luoda sekvenssejä.

# Sekvenssien luontia varten tulee tietää,
# kuinka monella edellisellä arvon avulla ennustetaan
# ja kuinka pitkälle tulevaisuuteen ennuste luodaan.
# Tässä tapauksessa voisimme ennustaa edellisen viikon arvoilla seuraavan päivän lämpötilat,
# jolloin datasetin ensimmäinen sekvenssi olisi:

# Arvot mitattu joka tunti, jolloin viikossa tunteja 24*7
x_seq = df[0:24*7].values
# Ennustetaan viikon jälkeisen päivän lämpötilat, eli datasetissä kahdeksas päivä
y_seq = df[24*7:24*8]['Temperature'].values
print(x_seq.shape)
print(y_seq.shape)
'''
(168, 2)
(24,)
'''
# Voimme käyttää myös aikaleima indeksiä tässä hyödyksi, jolloin voimme määrittää sekvenssin päivämäärillä.

x_seq = df['2016-01-01':'2016-01-07'].values
y_seq = df['2016-01-07']['Temperature'].values
print(x_seq.shape)
print(y_seq.shape)
'''
(168, 2)
(24,)
'''

# Luodaan tämän avulla funktio,
# jossa käymme toistorakenteessa datasetin läpi ja muokkaamme ominaisuuksista ja ennustettavista arvoista sekvenssejä.

# History parametrissä määritetään kuinka monella edeltävällä tunnilla ennustetaan ja forecast parametrissä kuinka monta tuntia tulevaisuuteen ennustetaan.
import numpy as np
def dataframe_to_sequences(df,history,forecast):
    x_sequences = []
    y_sequences = []
    i = 0
    while i < (len(df)):
        # Varmistetaan ettei hypätä indeksimuuttujassa Dataframen viimeisen indeksin yli
        if i+history+forecast < len(df):
            x_seq = df[i:i+history].values
            y_seq = df[i+history:i+history+forecast].values
        # lisätään sekvenssit listaan
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
        # Siirretään indeksiä 8 päivää eteenpäin, jolloin voimme luoda seuraavat sekvenssiparit
        i += history + forecast
    return np.array(x_sequences), np.array(y_sequences)
X, y = dataframe_to_sequences(df,24*7,24)
print(X.shape)
print(y.shape)
'''
(92, 168, 2)
(92, 24, 2)
'''
# Saimme siis 92 sekvenssiparia:
    # ominaisuuksia 168 tuntia, joilla ennustetaan
    # ja 24 tulevaa tuntia, joita ennustetaan.
    
#%%

# Lämpötilan ja sähkönkulutuksen arvot eroavat skaalassa paljon,
# on syytä skaalata nämä sarakkeet käyttäen MinMax - skaalausta, jotta neuroverkko voi määrätä painoarvot oikein.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(df.values)
# Datasetin arvot on nyt skaalattu välille [0-1].

#%% 

# TensorFlowin dokumentaatiossa on joustavuudeltaan parempi funktio,
# jonka avulla datasetin voi muuttaa sekvenssimuotoon
# sekä myös jakaa koulutus- ja testidataan samalla
# -> https://www.tensorflow.org/tutorials/structured_data/time_series.
# Alla kyseinen funktio sekä esimerkki sen käytöstä.

# Funktio, jonka avulla jaetaan data ominaisuuksiin ja ennustettaviin arvoihin
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
TRAIN_SPLIT = len(dataset) // 2 # Jaetaan datasetti kahteen, vuosi 2016 koulutusdataksi ja 2017 testidataksi
train_X, train_y = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
test_X, test_y = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
print (f'Single window of past history : {train_X[0].shape}')
print (f'Target to predict : {train_y[0].shape}')
'''
Single window of past history : (168, 2)
Target to predict : (24,)
'''

#%%

# Luodaan LSTM - neuroverkko ja koulutetaan se koulutusdatalla.

import tensorflow as tf
input_layer = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2])) # muoto on (None, 168, 2)
lstm1 = tf.keras.layers.LSTM(32,return_sequences=True)(input_layer)
lstm2 = tf.keras.layers.LSTM(16,activation='relu')(lstm1)
out = tf.keras.layers.Dense(future_target)(lstm2)
model = tf.keras.Model(inputs=input_layer,
                       outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse',metrics=['mse','mae'])
model.fit(train_X,train_y,validation_data=(test_X,test_y),epochs=6)

'''
Train on 8604 samples, validate on 8581 samples
Epoch 1/6
8604/8604 [==============================] - 28s 3ms/sample - loss: 0.2658 - mse: 0.2658 - mae: 0.4395 - val_loss: 0.1289 - val_mse: 0.1289 - val_mae: 0.2664
Epoch 2/6
8604/8604 [==============================] - 28s 3ms/sample - loss: 0.0800 - mse: 0.0800 - mae: 0.1988 - val_loss: 0.0347 - val_mse: 0.0347 - val_mae: 0.1265
Epoch 3/6
8604/8604 [==============================] - 28s 3ms/sample - loss: 0.0197 - mse: 0.0197 - mae: 0.0975 - val_loss: 0.0082 - val_mse: 0.0082 - val_mae: 0.0669
Epoch 4/6
8604/8604 [==============================] - 30s 4ms/sample - loss: 0.0065 - mse: 0.0065 - mae: 0.0615 - val_loss: 0.0046 - val_mse: 0.0046 - val_mae: 0.0514
Epoch 5/6
8604/8604 [==============================] - 29s 3ms/sample - loss: 0.0048 - mse: 0.0048 - mae: 0.0536 - val_loss: 0.0042 - val_mse: 0.0042 - val_mae: 0.0490
Epoch 6/6
8604/8604 [==============================] - 28s 3ms/sample - loss: 0.0046 - mse: 0.0046 - mae: 0.0521 - val_loss: 0.0041 - val_mse: 0.0041 - val_mae: 0.0481
<tensorflow.python.keras.callbacks.History at 0x2a7a67611c8>
'''

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

# Kuvaajasta näämme, ettei neuroverkkomalli vielä täysin oppinut ennustamaan yhden vuoden lämpötilojen
# ja ilmanpainearvojen perusteella ennustamaan seuraavan vuoden lämpötiloja.
# Kun lämpötila laskee tai nousee, neuroverkon ennustukset seuraavat 'viiveellä' perässä.

#%%

# Lähteet
# Aikasarjaennustaminen Kerassilla https://www.tensorflow.org/tutorials/structured_data/time_series