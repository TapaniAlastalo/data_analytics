from IPython.display import Image
import tensorflow as tf

# Konvoluutiokerros määritetään 2D - matriiseille Kerassissa vastaavasti:
#  Konvoluutiokerrokselle määritetään parametrit:

# filters = Kuinka monta feature matriisia konvoluutiokerros luo
    # Tämä on samantapainen kuin neuronien määrä Dense - kerroksissa. 
    # Jokainen luotu feature matriisi on erilainen, koska jokaisella filtterillä on eri painoarvot
    # (eli tässä tapauksessa arvot kernel - matriisissa)
# kernel_size = Kuinka suuri kernel matriisi on, jonka avulla konvoluutiossa lasketaan uudet arvot
# strides = Kuinka monta askelta siirrytään jokaisen laskun jälkeen
# padding = Lisätäänkö matriisin reunoille arvoja, jotta saadaan ulostulomatriisin koko parillisena vai ei. 
    # Arvo 'same' lisää nollia matriisin reunoille (zero padding), kun 'valid' ei.
    
# tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same')

(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()
print(train_X.shape)
# (60000, 28, 28) # eli 60 000 kpl kuvia, joiden koko 28x28

#%%
# MNIST = mustavalkoinen -> värikanava 1
# Muokataan MNIST kuvien muoto muotoon: (kuvien määrä, koko pystysuunnassa, koko vaakasuunnassa, värikanavien määrä)
train_X = train_X.reshape((train_X.shape[0],28,28,1))

# Jos padding on 'same', niin matriisia, johon konvoluutio suoritetaan, täytetään nollilla, niin että ulostulossa on samat dimensiot (leveys ja korkeus) kuin sisääntulossa.
# Tämä pätee vain, jos askel on yksi
model_input = tf.keras.Input(shape=(28,28,1)) 
# konvoluutiokerros
model_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=1, padding='same')(model_input)
model = tf.keras.Model(inputs=model_input,
                        outputs=model_conv)
print(model.output.shape)
# TensorShape(None, 28, 28, 32)

# Jos padding 'same' muutetaan 'valid' arvoon, ulostulomatriisin koko ei ole enää sama kuin sisääntulon
model_input = tf.keras.Input(shape=(28,28,1)) 
model_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=1, padding='valid')(model_input)
model = tf.keras.Model(inputs=model_input,
                        outputs=model_conv)
print(model.output.shape)
# TensorShape([None, 27, 27, 32])
# Ulostulossa tensorissa 32 kpl 28x28 matriisia. Askeleen kasvatus pienentää feature matriiseja.

# Askeleen kohdalle voi määrittää kaksi kokonaislukua tuple - muodossa, niin kuin kernel_size parametrissä on määritelty.
# Tuplen ensimmäinen arvo kertoo kuinka monta askelta liikutaan kerralla vaakasuunnassa,
# toinen arvo taas sen kuinka monta askelta siirrytään pystysuunnassa, kun matriisi on käyty vaakasuunnassa läpi.
model_input = tf.keras.Input(shape=(28,28,1)) 
model_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=(2,1), padding='same')(model_input)
model = tf.keras.Model(inputs=model_input,
                        outputs=model_conv)
print(model.output.shape)
# TensorShape([None, 14, 28, 32])

model_input = tf.keras.Input(shape=(28,28,1)) 
model_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=(2,3), padding='same')(model_input)
model = tf.keras.Model(inputs=model_input,
                        outputs=model_conv)
print(model.output.shape)
# TensorShape([None, 14, 10, 32])
# askeleen pituuden muutos pystysuunnassa; ulostulon korkeus muuttui 28 -> 10


#%%
# Konvoluution ulostulon laskukaavat
# Luodaan vielä konvoluutio- ja Max Pooling kerrokset Kerassissa ja tulostetaan kummankin operaation ulostulojen muodot.
model_input = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2],1)) 
# konvoluutiokerros
model_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=1, padding='same')(model_input)
model_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(model_conv)
model = tf.keras.Model(inputs=model_input,
                        outputs=model_maxpool)
print(model.layers[1].output)
print(model.output)

# Tensor("conv2d_6/BiasAdd:0", shape=(None, 28, 28, 32), dtype=float32)
# Tensor("max_pooling2d/MaxPool:0", shape=(None, 14, 14, 32), dtype=float32)
# Huomataan, että ensimmäinen konvoluutio säilytti sisääntulomatriisin 28x28 muodon samana,
# mutta MaxPooling kerros puolitti feature - matriisien koot 14x14 kokoiseksi.


#%%
# Filter- ja feature - matriisien visualisointi

# Yleensä on vaikea hahmottaa, mitä neuroverkkojen sisällä oikeasti tapahtuu.
# Konvoluutioneuroverkoissa tosin on mahdollista visualisoida feature - matriisit,
# joita syntyy konvoluutio-operaatioiden tuloksena.
# Näin voimme saada jotain ideaa siitä, mitä ominaisuuksia konvoluutioverkon avulla voidaan tunnistaa kuvista.

# Filter - matriisi on yhdistelmä kernel - matriiseja, ja ne ovat yhden dimension laajempia kuin kernelit.
# Eli kun tehdään 2D konvoluutioita, filter - matriisit olisivat 3D - matriiseja eli yhdistelmä 2D kernel - matriiseja.

# Ladataan Kerassilla valmis VGG16 malli, koska sen rakenne on helppo ymmärtää
# ja sillä on saatu kuvantunnistuksissa myös hyviä tuloksia.

# lataa painoarvot mallille, jotka ovat n. 500 megatavua.
#vgg16 = tf.keras.applications.VGG16()
#vgg16.summary()

# Näemme model.summary() funktion tulosteesta, minkälainen on VGG16 neuroverkon rakenne.
# VGG16 malli sisältää viisi konvoluutiolohkoa, jossa suoritetaan kaksi konvoluutiota ja yksi Max Pooling - operaatio.

#%%
# Filter - matriisit

# Filter - matriisit toimivat kuten yhteyksien väliset painoarvot Dense - neuroverkkokerroksissa.
# Voimme visualisoida, miltä VGG16 neuroverkossa ensimmäisen konvoluution filter - matriisi näyttää
import matplotlib.pyplot as plt
# Hae ensimmäisen kerroksen filter - matriisi
filters, _ = vgg16.layers[1].get_weights()
# normalisointi
# Filter - matriisit ovat 3x3 kokoisia, piirretään kolme ensimmäisestä värikanavasta. VGG16 neuroverkko tunnistaa värikuvia, joten värikanavia on yhteensä 3.
fig, axs = plt.subplots(1,3)
f = filters[:, :, :, 0]
for i in range(3):
    axs[i].imshow(f[:,:,i],cmap='gray')
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.show()

# Kuvassa valkoinen väri tarkoittaa suurempaa painoarvoa kuin musta. 
# Tässä filtterit tunnistavat, miten kuvan värit muuttuvat vasemmalta ylhäältä oikealle alas,
# jonka avulla voisi tunnistaa esim. varjoja tai eläimen silmät.

#%%
# Feature matriisit

# Kun sisääntulevaan kuvaan suoritetaan konvoluutio käyttäen filter - matriiseja, saadaan tuloksena feature - matriiseja.
# Feature matriiseja syötetään VGG16 neuroverkossa eteenpäin seuraavaan konvoluutio-operaatioon.

# Jos piirretään kyseiset feature - matriisit, voimme nähdä, miten sisääntuleva kuva muuttuu eri konvoluutiokerrosten jälkeen.
# Summary - funktion tuloksesta näämme, että feature - matriisien koot pienentyvät jokaisen Max Pooling - kerroksen jälkeen.
# Tällöin feature - matriisit sisältävät alussa paljon yksityiskohtia kuvasta,
# mutta loppua feature - matriisit keskittyvät kuvan pääominaisuuksiin,
# kuten kuvassa esiintyvien asioiden reunoihin.

# Syötetään VGG16 mallille kuva puumasta ja visualisoidaan ensimmäisen konvoluutiokerroksesta ulostuleva feature - matriisit.
# Summary - funktiosta nähdään, että ensimmäisen konvoluutiokerroksen ulostulo on 64 feature - matriisia,
# joiden leveydet ja korkeudet ovat 224.

import numpy as np
# Käsittele kuva käyttäen Kerasin valmiita funktioita
def preprocess_img(img):
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img
# Luo osamalli VGG16:sta, jossa ulostulo on ensimmäisen konvoluutiokerroksen ulostulo.
vgg16_1 = tf.keras.Model(inputs=vgg16.inputs,
             outputs=vgg16.layers[1].output)
# Lataa kuva
img = tf.keras.preprocessing.image.load_img('mountain_lion.jpg',target_size=(224,224))
img = preprocess_img(img)
# Luo feature - matriisit kuvasta
feature_maps = vgg16_1.predict(img)
# Visualisoi feature - matriisi
rows, cols = 8,8
fig, axs = plt.subplots(rows, cols,figsize=(20,10))
index = 0
for i in range(rows):
    for j in range(cols):
        axs[i,j].imshow(feature_maps[0,:,:,index],cmap='gray')
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])
        index += 1
plt.show()

# Jos haluamme nähdä feature mapsit jokaisesta viidestä lohkosta,
# tulee Model muuttujaan ulostuloihiun määritellä useamman kerroksen ulostulo

# lohkojen ulostulojen indeksit
block_outputs_index = [2, 5, 9, 13, 17]
# Hae ulostulot lohkoista
block_outputs = [vgg16.layers[i].output for i in block_outputs_index]
# Luo malli VGG16:sta, jossa on ulostulona useampi lohkon ulostulo
vgg16_2 = tf.keras.Model(inputs=vgg16.inputs, outputs=block_outputs)
# Luo feature - matriisit kuvasta
feature_maps = vgg16_2.predict(img)
# Visualisoi kaikkien lohkojen feature - matriisit.
rows, cols = 8,8
block_counter = 1
for feature_map in feature_maps:
    fig, axs = plt.subplots(rows, cols,figsize=(20,10))
    index = 0
    for i in range(rows):
        for j in range(cols):
            axs[i,j].imshow(feature_map[0,:,:,index],cmap='gray')
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            index += 1
    print(f'Block {block_counter} Feature matriisi')
    block_counter += 1
    plt.show()

#%%
# Feature maps visualisointi https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/