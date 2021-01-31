import tensorflow as tf

# Kerassissa regularisaation voi määritellä kerroksissa kerneliin, biasiin ja ulostuloon:
dense_regularized = tf.keras.layers.Dense(
    10,
    kernel_regularizer = tf.keras.regularizers.l2(0.001), # Regularisaatiofunktio painoarvoihin
    bias_regularizer = tf.keras.regularizers.l1(0.01), # Regularisaatiofunktio biassiin
    activity_regularizer = tf.keras.regularizers.l2(0.01) # Regularisaatiofunktio aktivaatiofunktioon
)
dense_regularized

#%%
# Dropout

# Dropout regularisaatiotekniikassa osa kerroksen ulostuloista merkitään nollaksi.
# Tämä pätee, kun neuroverkon avulla luodaan ennuste ja kun neuroverkon painoarvoja päivitetään.
# Ideana on se, että neuronit eivät luota muiden neuronien ulostuloihin liikaa.
# Esim. jokin neuroni saattaa aktivoitua, kun syötössä on ominaisuus x.
# Jos tämä neuroni koulutuksen aikana satunnaisesti deaktivoidaan, niin neuroverkon täytyy mukautua niin,
# että myös muut neuronit aktivoituvat, kun syötössä on ominaisuus x.

# Yleensä neuroneista deaktivoidaan 20-50% ylioppimisen estämiseksi.
# Kerassissa Dropout määritellään kuin mikä tahansa muu neuroverkon kerros.
# Dropout tulee määritellä sen kerroksen jälkeen, mistä haluaa deaktivoida neuroneita

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=5)
"""
Jos koulutettaisiin:
model.fit(
    train_X,
    train_y,
    epochs=50,
    callbacks=[callback],
    validation_data=(test_X,test_y)
)
"""
#%%
# Ylioppiminen ja regulaatio

# Luodaan ensin tarkoituksella malli, joka ylioppii. Ylioppineen mallin huomaa vertaamalla koulutus- ja testivirhettä.
# Jos testivirhe on paljon suurempi kuin koulutusvirhe, se tarkoittaa että malli on mukautunut liian tarkasti koulutusdataan.
# Kun ylioppinut malli on luotu, lisätään malliin regularisaatiomenetelmiä,
# jonka jälkeen verrataan taas koulutus- ja testivirhettä.

# Luodaan oma datasetti käyttämällä scikit-learn kirjaston "make_moons" - funktiota,
# joka luo pisteitä kahteen eri puoliympyrän muotoon. Visualisoidaan luotu datasetti kuvaajaan.
# Jotta saamme pisteiden leimat näkymään, käytämme seaborn - kirjaston FacetGrid luokkaa apuna.

from sklearn.datasets import make_moons
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks')

dots,labels = make_moons(n_samples=1000, noise=0.30, random_state=1)
print(dots.shape,labels.shape)
df = pd.DataFrame(data={'x':dots[:,0], 'y':dots[:,1], 'Label':labels})

# Kopioitu vastauksesta https://stackoverflow.com/questions/14885895/color-by-column-values-in-matplotlib
fig = sns.FacetGrid(data=df, hue='Label', hue_order=df['Label'].unique(), aspect=2)
fig.map(plt.scatter, 'x', 'y').add_legend()
# Kyseessä on epälineaarinen ongelma, sillä pisteitä ei voi jakaa hyvin kahteen osaan piirtämällä suoraa viivaa.

#%%
# Luodaan ensin Kerassilla neuroverkko, jossa tapahtuu ylioppimista.
# Tämä onnistuisi esim. luomalla liian suuren neuroverkon, jossa epookkien määrä olisi myös suuri

# Suoritetaan koulutus- ja testidata jako. Laitetaan jaon suhteeksi 20/80
train_X, train_y = dots[:len(dots) // 8, :], labels[:len(dots) // 8]
test_X, test_y = dots[len(dots) // 8:, :], labels[len(dots) // 8:]
# Käytetään Sequential mallia, koska luomme yksinkertaisia neuroverkkoja
model_overfit = tf.keras.Sequential([
    tf.keras.layers.Dense(500,activation='relu',input_shape=(2,)),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
opt = tf.keras.optimizers.Adam(lr=0.1)
model_overfit.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
# Kirjataan koulutus- ja testitarkkuus koulutuksen aikana määrittelemällä validation_data parametri. 
history_overfit = model_overfit.fit(train_X,
                train_y,
                validation_data = (test_X,test_y),
                batch_size = 2**5,
                epochs=1000,
                verbose=0)
# Piirretään tarkkuudet koulutuksen aikana graafiin
plt.plot(history_overfit.history['accuracy'],label='train')
plt.plot(history_overfit.history['val_accuracy'],label='test')
plt.legend()
plt.show()

#%%
# Luodaan seuraavaksi malli, jossa käytetään L2 regularisaatiota Densen kernelissä.
model_l2 = tf.keras.Sequential([
    tf.keras.layers.Dense(500,activation='relu',input_shape=(2,),kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model_l2.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
history_l2 = model_l2.fit(train_X,
                train_y,
                validation_data = (test_X,test_y),
                batch_size = 2**5,
                epochs=1000,
                verbose=0)
plt.plot(history_l2.history['accuracy'],label='train')
plt.plot(history_l2.history['val_accuracy'],label='test')
plt.legend()
plt.show()
# Huomataan, että testitarkkuus on paljon lähempänä koulutustarkkuutta, kun Dense kerrokseen lisättiin L2 regularisaatio.

#%%
# Kokeillaan seuraavaksi lisätä Dropout - kerros Densen jälkeen
model_dropout = tf.keras.Sequential([
    tf.keras.layers.Dense(500,activation='relu',input_shape=(2,)),
    tf.keras.layers.Dropout(0.5), # Deaktivoidaan puolet neuroneista
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model_dropout.compile(optimizer=opt
                            ,loss='binary_crossentropy',
                            metrics=['accuracy'])
history_dropout = model_dropout.fit(train_X,
                train_y,
                validation_data = (test_X,test_y),
                batch_size = 2**5,
                epochs=1000,
                verbose=0)
plt.plot(history_dropout.history['accuracy'],label='train')
plt.plot(history_dropout.history['val_accuracy'],label='test')
plt.legend()
plt.show()

# Ylioppineessa mallissa nähdään kuvaajasta, että testitarkkuus saavuttaa huippunsa muutamassa epookissa,
# mutta koulutuksen jatkuessa testitarkkuus pienenee ja koulutustarkkuus suurenee.
# Malleissa, joissa on implementoitu L2 ja Dropout - regularisaatiomenetelmiä,
# koulutus- ja testitarkkuus pysyvät keskimäärin huippuarvossaan koulutuksen ajan.

#%%
# L1 ja L2 regularisaatiot https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
# Keras ylioppiminen esimerkki https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/