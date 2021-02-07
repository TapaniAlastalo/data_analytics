# Tehtävän toteutus

# Lataa luottokorttihuijaus datasetti
import pandas as pd
df = pd.read_csv('c:/data/creditcard.csv')
#print(df)

#df = df.drop(columns=['Time'], axis=1)
df = df.drop(columns=['Time','Amount','Class'], axis=1)
#clss = df['Amount'] #['Class']
#df = df.iloc[:,1:-2]
print(df)

# Jaa datasetti koulutus- ja testidatasettiin
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)#, stratify=clss)

# skaalataan arvot
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)


#%%
import tensorflow as tf
# Luo autoenkooderimalli, jossa sisääntulevasta luottokorttitapahtumatiedoista tiivistetään pienin mahdollinen esitys,
features = train.shape[1] # Kuinka monta saraketta datasetissä
input_layer = tf.keras.Input(shape=(features,))
encoder = tf.keras.layers.Dense(7,activation='relu')(input_layer) # Enkooderi, antaa ulos pienemmän esityksen syötteestä, eli vektori z aiemmassa kuvatussa pullonkaulassa
decoder = tf.keras.layers.Dense(features, activation='sigmoid')(encoder) # Dekooderi
autoencoder = tf.keras.Model(inputs=input_layer,
                             outputs=decoder)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='mean_squared_error')
autoencoder.fit(train, train,epochs=20,batch_size=4)

#%%
import numpy as np
test_predictions = autoencoder.predict(test)
mse_test = np.mean((test - test_predictions)**2)
print(mse_test)

train_predictions = autoencoder.predict(train)
mse_train = np.mean((train - train_predictions)**2)
print(mse_train)

# Määritä koulutusdatan avulla raja-arvo virheelle, minkä ylittävät tapahtumat luokitellaan luottokorttihuijauksiksi

# Tutki numeerisesti tai visuaalisesti, kuinka hyvin raja-arvolla tunnistetaan poikkeamat testidatasetistä


#%%

# Tehtävän vastaukset. Käytä visualisaatiota tai osoita numeerisesti, kuinka monta poikkeamaa neuroverkkomalli huomasi.
import matplotlib.pyplot as plt
test_predictions = autoencoder.predict(test)
mse_test = np.mean((test - test_predictions)**2,axis=1)
trained_predictions = autoencoder.predict(train)
mse_trained = np.mean((train - trained_predictions)**2,axis=1)
predictions_mse = np.concatenate([mse_test,mse_trained]) # Yhdistetään ennustukset yhteen taulukkoon kuvaajaa varten
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(range(len(predictions_mse)),predictions_mse)
ax.scatter(range(len(mse_test)),mse_test)