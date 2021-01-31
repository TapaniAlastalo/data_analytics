# Valmiiksi koulutetut neuroverkot

from IPython.display import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG16
vgg16 = VGG16()
vgg16.summary()

#%%
# Johtuen neuroverkon rakenteesta, VGG16 tarvitsee syötteenä vähintään 32x32 kokoisia värikuvia.
# Tarkastetaan, että CIFAR10 kuvat ovat vähintään 32x32 kookisia.
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.cifar10.load_data()
print(train_X.shape)
# (50000, 32, 32, 3)
# CIFAR10 datasetissä on koulutusdatassa 50000 kuvaa, jotka ovat 32x32 kokoisia ja joissa on kolme värikanavaa (R,G,B).

#%%
# Luodaan VGG16 neuroverkko uudestaan, mutta laitetaan tällä kertaa parametreiksi uuden kuvien muoto
# ja poistetaan viimeiset klassifikaatiokerrokset 'include_top' parametrillä.
# Klassifikaatiokerrokset ovat viimeiset kaksi Dense - kerrosta Flatten - operaation jälkeen.
# Konvoluutiokerrokset toimivat VGG16:ssa siis ominaisuuksien etsijinä kuvista
# ja klassifikaatiokerrokset määrittävät sen avulla, mikä esine kuvassa on.
vgg16 = VGG16(input_shape=(32,32,3),include_top=False)
vgg16.summary()
# Total params: 14, 714 688

#%%
# Seuraavaksi luodaan uudestaan viimeiset Dense - kerrokset ja ulostulokerros.
# Luodaan siis yhteensä kolme Dense - kerrosta.
# Aikaisemmasta Summary - funktion tuloksesta huomattiin,
# että myös Flatten - kerros oli poistunut include_top parametrin takia, joten luodaan sekin uudestaan.

flatten = tf.keras.layers.Flatten()(vgg16.output)
new_dense1 = tf.keras.layers.Dense(512,activation='relu')(flatten)
new_dense2 = tf.keras.layers.Dense(512,activation='relu')(new_dense1)
new_out = tf.keras.layers.Dense(10,activation='softmax')(new_dense2) # 10 eri classia, joiden todennäköisyyksiä ennustetaan

#%%

# Liitetään nämä kerrokset vanhojen konvoluutiokerrosten jatkoksi.
# Jotta koulutusvaiheessa koko neuroverkkoa ei opetettaisi uudestaan vaan pelkästään klassifikaatiokerrokset,
# niin konvoluutiokerrokset voi jäädyttää muuttamalla 'trainable' parametrin 'False':ksi.
# Näin koulutuksen backpropagation vaiheessa konvoluutiokerrosten painoarvoja ei päivitetä, mikä säästää paljon laskentatehoa.
for layer in vgg16.layers:
    layer.trainable = False
vgg16_cifar = tf.keras.Model(inputs=vgg16.input,
                             outputs=new_out)
vgg16_cifar.summary()
# Total params: 15,245,130
# Trainable params: 530,442
# Non-trainable params: 14,714,688


#%%
# Siirrytään nyt itse siirto-oppimiseen, eli koulutetaan VGG16:n konvoluutiokerrosten valmiilla painoarvoilla
# ratkaisemaan CIFAR10 datasetin kuvantunnistusongelma.
vgg16_cifar.compile(loss='sparse_categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                    metrics=['accuracy'])
vgg16_cifar.fit(train_X,
                train_y,
                epochs=10)

# Mitataan koulutetun mallin tarkkuus testidatasetin avulla
results = vgg16_cifar.evaluate(test_X,
                               test_y,
                               verbose=0)
print(f"Test loss: {results[0]} Test accuracy: {results[1] * 100:.0f}%")
# Test loss: 1.660546898841858 Test accuracy: 60%

#%%
# Verkon määritys ja paniarvojen lataus manuaalisesti
import numpy as np 
import matplotlib.pyplot as plt

X = np.arange(0,10) # Arvot väliltä 0-9
y = (2*X) + 0.1 + 0.5*np.random.randn(10) # Funktio: y = 2x + 1/10 + kohinaa
plt.plot(y)
plt.show()

# ihanasti nousee

#%%
import tensorflow as tf
# Sisääntulodatan muodon muokkaus
X = X.reshape((10,1))
# Mallin luonti
input_layer = tf.keras.Input(shape=(1,))
dense1 = tf.keras.layers.Dense(12,activation='relu')(input_layer)
dense2 = tf.keras.layers.Dense(6,activation='relu')(dense1)
output_layer = tf.keras.layers.Dense(1,activation='linear')(dense2)
model = tf.keras.Model(inputs=input_layer,
                       outputs=output_layer)
# Koulutus
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
model.fit(X,y,epochs=500,verbose=0)
# Tulokset
test_X = np.linspace(0,10,100) # Luodaan testidatana lineaarisesti 100 arvoa väliltä 0-10
preds = model.predict(test_X)
plt.plot(test_X,preds)

# suora!

#%%

# Tallenetaan neuroverkon painoarvot tiedostoon
model.save_weights('linear_regression_weights.h5')

# Äsken luodun neuroverkon painoarvot ovat nyt tiedostossa 'linear_regression_weights.h5'.
# Jotta painoarvot voi ladata uudestaan toiseen malliin, tulee neuroverkon rakenne tietää täysin,
# eli missä järjestyksessä kerrokset ovat ja kuinka monta neuronia niissä on.

#%%
#Kuvitellaan, että emme tiedä millä koodilla äskeinen malli luotiin, vaan tehdään se alla olevan kuvan perusteella:

import os
# Graphviz exe:n polun määritys
os.environ['PATH'] = os.environ['PATH']+';'+ os.environ['CONDA_PREFIX'] + r"\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz"
# Neuroverkon rakenne kuvaksi
tf.keras.utils.plot_model(model,dpi=70,show_shapes=True)

#%%
model2 = tf.keras.Sequential([
# Lähdetään luomaan neuroverkon rakennetta uudestaan, tällä kertaa käyttäen vaikka Sequential - luokkaa.
    # Jätetään InputLayer pois ja korvataan se määrittelemällä input_shape - parametri ensimmäiseen Dense kerrokseen
    tf.keras.layers.Dense(12, input_shape=(1,),activation='relu'), # Tarvittava neuronien määrä nähdään kuvasta "output" - kentästä
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1)
])
model2.load_weights('linear_regression_weights.h5') # Ladataan painoarvot tiedostosta malliin
model2.layers[0].get_weights() # Tulostetaan ensimmäisen Dense - kerroksen painoarvot ja biakset


#%%
# Tehdään sama testi, kuin aikaisemmin
test_X = np.linspace(0,10,100)
preds = model2.predict(test_X)
plt.plot(test_X,preds)

# Onnistuimme tekemään saman mallin luomalla neuroverkon rakenteen uudestaan kuvan avulla
# ja tuomalla painoarvot tiedostosta.

#%%

