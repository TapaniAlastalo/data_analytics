from IPython.display import Image
import os
os.environ['PATH'] = os.environ['PATH']+';'+ os.environ['CONDA_PREFIX'] + r"\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz"

# Ohjaamaton oppiminen
# Kurssin alussa mainittiin, että ohjatussa oppimisessa neuroverkkoa koulutetaan label-feature pareilla,
# eli neuroverkolle annetaan syötearvot ja neuroverkon virhe lasketaan siitä,
# kuinka paljon neuroverkon ennuste erosi halutusta lopputuloksesta,
# jonka avulla neuroverkon painoarvoja muokataan, kunnes neuroverkon ennuste lähenee haluttua lopputulosta.
# Useimmissa tapauksissa valtavalle datamäärälle halutun lopputuksen määrittäminen on kuitenkin työlästä
# tai joissakin tapauksissa datasetistä halutaan poimita poikkeuksia tai tiivistää alkuperäinen data pienempään muotoon.
# Näissä tapauksissa neuroverkkoja voidaan kouluttaa myös ohjaamattoman oppimisen menetelmällä.

# Kun ohjaamattomassa oppimisessa ei ole ollenkaan määrätty haluttua lopputulosta,
# josta neuroverkon ennusteelle voitaisiin laskea virhe, täytyy neuroverkon ennustuksen hyvyyttä mitata eri tavalla.
# Yksi tapa olisi esimerkiksi mitata, kuinka lähellä neuroverkon ennuste on neuroverkon sisääntuloa,
# jos tarkoituksena olisi esim. poikkeamien tunnistaminen.
# Kun neuroverkon ennuste eroaa tietyn raja-arvon verran yli sisääntulosta, niin todettaisiin,
# että sisääntuloa poikkeaa siitä datasta, millä neuroverkko on opetettu.

# Autoenkooderi
# Autoenkooderi on neuroverkkomalli, joka koostuu enkooderista ja dekooderista.
# Autoenkooderit koulutetaan ohjaamattomalla oppimisella, eli datasetissä ei ole feature-label pareja.
# Autoenkooderin tehtävänä on ottaa syöte, luoda siitä enkoodattu versio
# ja rakentaa syöte uudelleen enkoodatusta tiedosta.
# Tarkoituksena on saada ulostulosta mahdollisimman paljon sisääntuloa muistuttava tulos.

# Autoenkooderin komponentit ovat:

# Enkooderi - malli, jolla muodostetaan syöttödatasta mahdollisimman tiivis esitys
# Pullonkaula - kerros, jossa tiivistetty esitys sijaitsee
# Dekooderi - malli, joka luo tiivistetystä esityksestä mahdollisimman paljon syöttödataa imitoivan version
# Virhefunktio, joka mittaa, kuinka hyvin autoenkooderin ulostulo vastaa sisääntuloa
# Autoenkooderissa siis enkooderi pakkaa sisääntulon pienempään muotoon
# ja dekooderi yrittää päätellä, miltä data näytti ennen pakkausta.
#  Autoenkooderin ulostulo tulisi olla aina hieman erilainen kuin sisääntulo,
# muuten enkooderi ei ole onnistunut pakkaamaan dataa onnistuneesti tiiviimpään muotoon.

#%%

# Poikkeamien tunnistus
# Kun autoenkooderi koulutetaan, se oppii poimimaan koulutusdatasetin sisääntuloista ominaisuudet,
# joiden avulla parhaiten rakentaa sisääntulon tiivistetystä esityksestä sisääntulo uudestaan.
# Jos autoenkooderille syötettäisiin jokin uusi sisääntulo,
# joka poikkeaisi autoenkooderin oppimasta rakenteesta,
# niin autoenkooderi ei osaisi luoda sisääntuloa uudelleen samannäköiseksi.
# Tämän avulla autoenkooderia voisi käyttää poikkeaman tunnistukseen.

# Kokeillaan yksinkertaisena esimerkkinä kouluttaa autoenkooderille
# Boston asuntojen hinnat - datasetin sisääntuloarvot.
# Kun autoenkooderi on koulutettu,
# syötetään sille datasetin testiarvoja sekä muokattuja testiarvoja.

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
tf.keras.backend.set_floatx('float64')
boston = tf.keras.datasets.boston_housing
(train_X, _), (test_X, _) = boston.load_data()

# skaalataan arvot
scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# valitaan 50 muokkaamatonta arvoa
test_X_normal = test_X[:50]

# muokataan loput arvot kertomalla ne kymmenellä
test_X_modified = test_X[50:] * 10
test_X_modified = scaler.transform(test_X_modified)

# Luodaan autoenkooderi
features = train_X.shape[1] # Kuinka monta saraketta datasetissä
input_layer = tf.keras.Input(shape=(features,))
encoder = tf.keras.layers.Dense(7,activation='relu')(input_layer) # Enkooderi, antaa ulos pienemmän esityksen syötteestä, eli vektori z aiemmassa kuvatussa pullonkaulassa
decoder = tf.keras.layers.Dense(features, activation='sigmoid')(encoder) # Dekooderi
autoencoder = tf.keras.Model(inputs=input_layer,
                             outputs=decoder)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='mean_squared_error')
autoencoder.fit(train_X,train_X,epochs=20,batch_size=4)

'''
Epoch 1/20
  1/101 [..............................] - ETA: 0s - loss: 0.1493WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0000s vs `on_train_batch_end` time: 0.0010s). Check your callbacks.
101/101 [==============================] - 0s 475us/step - loss: 0.1277
Epoch 2/20
101/101 [==============================] - 0s 455us/step - loss: 0.1014
Epoch 3/20
101/101 [==============================] - 0s 455us/step - loss: 0.0781
Epoch 4/20
101/101 [==============================] - 0s 455us/step - loss: 0.0610
Epoch 5/20
101/101 [==============================] - 0s 465us/step - loss: 0.0501
Epoch 6/20
101/101 [==============================] - 0s 465us/step - loss: 0.0429
Epoch 7/20
101/101 [==============================] - 0s 455us/step - loss: 0.0379
Epoch 8/20
101/101 [==============================] - 0s 455us/step - loss: 0.0343
Epoch 9/20
101/101 [==============================] - 0s 446us/step - loss: 0.0318
Epoch 10/20
101/101 [==============================] - 0s 448us/step - loss: 0.0299
Epoch 11/20
101/101 [==============================] - 0s 394us/step - loss: 0.0283
Epoch 12/20
101/101 [==============================] - 0s 395us/step - loss: 0.0269
Epoch 13/20
101/101 [==============================] - 0s 497us/step - loss: 0.0256
Epoch 14/20
101/101 [==============================] - 0s 497us/step - loss: 0.0245
Epoch 15/20
101/101 [==============================] - 0s 493us/step - loss: 0.0235
Epoch 16/20
101/101 [==============================] - 0s 496us/step - loss: 0.0225
Epoch 17/20
101/101 [==============================] - 0s 393us/step - loss: 0.0217
Epoch 18/20
101/101 [==============================] - 0s 399us/step - loss: 0.0209
Epoch 19/20
101/101 [==============================] - 0s 393us/step - loss: 0.0201
Epoch 20/20
101/101 [==============================] - 0s 399us/step - loss: 0.0193
'''

#%%

# Luodaan autoenkooderilla ennustukset yhdelle muokkaamattomalle testiarvolla ja muokatulle testiarvolle

normal_predictions = autoencoder.predict(test_X_normal[0:1])
modified_predictions = autoencoder.predict(test_X_modified[0:1])

# Lasketaan 'reconstruction error', eli kuinka paljon autoenkooderin ennuste eroaa syötteestä.
# Käytetään samaa virheen kaavaa kuin autoenkooderin opetuksessa, eli keskineliövirhettä

import numpy as np
mse_normal = np.mean((test_X_normal[0:1] - normal_predictions)**2)
print(mse_normal)
'''
0.03363559090746348
'''

mse_modified = np.mean((test_X_modified[0:1] - modified_predictions)**2)
print(mse_modified)
'''
0.21336289145127688
'''
# Näämme, että virhe on huomattavasti suurempi muokatun syötteen ennustuksessa,
# eli autoenkooderi osaa erottaa poikkeamat sille annetusta syötteestä.

#%%

# Käytännön toteutuksessa tulisi tarkastella,
# kuinka suuri virhe syötteillä yleisesti on ja mikä on virheen keskihajonta,
# joiden avulla virheelle määritettäisiin raja-arvo,
# jonka ylittyessä syöte luokiteltaisiin poikkeamaksi.
# Luodaan ennustukset kaikille testiarvoilla ja visualisoidaan virhearvot pisteinä kuvaajaan,
# jolloin näämme paremmin, miten ne jakautuvat.

import matplotlib.pyplot as plt
normal_predictions = autoencoder.predict(test_X_normal)
mse_normal = np.mean((test_X_normal - normal_predictions)**2,axis=1)
modified_predictions = autoencoder.predict(test_X_modified)
mse_modified = np.mean((test_X_modified - modified_predictions)**2,axis=1)
predictions_mse = np.concatenate([mse_normal,mse_modified]) # Yhdistetään ennustukset yhteen taulukkoon kuvaajaa varten
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(range(len(predictions_mse)),predictions_mse)

# Kuvaajasta näämme, että muokkaamattomat ensimmäiset 50 syötteen virhearvot pysyvät nollan tuntumassa,
# kun taas lopuissa muokatuissa syötteissä virhearvot vaihtelevat reilusti.
# Kuvaajan perusteella raja-arvo virhearvolle voisi olla esimerkiksi 5,
# jolloin osa poikkeamista löydettäisiin syötteistä.

# Määrittelemällä virheen raja-arvo liian matalaksi aiheuttaa monta väärää hälytystä poikkeamista,
# kun taas liian korkea raja-arvo jättää huomioimatta useamman poikkeavan arvon.

#%%

# MNIST kuvien uudelleenluonti Kerasilla
# Helpoin tapa visualisoida autoenkooderin toiminta on syöttää sille kuvia ja luoda ne uudelleen saman muotoisena,
# mutta hieman erilaisena. Ladataan tätä varten MNIST - datasetistä koulutuskuvia.

import tensorflow as tf
(images, _), (_, _) = tf.keras.datasets.mnist.load_data()
# normalisoidaan data välille 0 - 1
images = images / 255.0
# lisätään värikanavaa kuvaava sarake loppuun, jotta kuvat voi syöttää neuroverkolle
images = images.reshape(images.shape[0],28,28,1)
print(images.shape)
'''
(60000, 28, 28, 1)
'''

#%%

# Luodaan ensin enkooderi. Tämä näyttää paljon tavalliselta konvoluutioneuroverkolta,
# mutta sen sijaan että viimeisessä Dense kerroksessa luotaisiin ennuste kuvasta,
# luodaan mahdollisimman tiivistetty luonnos kuvasta.
# Tämä onnistuu Flatten() ja Dense kerroksella,
# jolloin viimeiseen Dense kerrokseen laitetaan aktivaatiofunktioksi 'linear'.

enc_input = tf.keras.Input(shape=(28,28,1))
enc_conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=(2,2),activation='relu')(enc_input)
enc_conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(2,2),activation='relu')(enc_conv1)
enc_flatten = tf.keras.layers.Flatten()(enc_conv2)
enc_out = tf.keras.layers.Dense(64,activation='linear')(enc_flatten)
encoder = tf.keras.Model(inputs=enc_input,
                         outputs=enc_out,
                         name='Encoder')
encoder.summary()

'''
Model: "Encoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_12 (InputLayer)        [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 13, 13, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 6, 6, 64)          18496     
_________________________________________________________________
flatten_3 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_18 (Dense)             (None, 64)                147520    
=================================================================
Total params: 166,336
Trainable params: 166,336
Non-trainable params: 0
'''

#%%

# Luodaan seuraavaksi dekooderi. Dekooderin tehtävä on luoda tiivistetty tieto takaisin kuvaksi.
# Tätä varten tarvitaan Conv2DTranspose - kerrosta Kerassissa,
# jonka avulla pienempi sisääntulo muutetaan suuremmaksi sisääntuloksi.
# Tätä kutsutaan myös dekonvoluutioksi. Näin saamme muodostettua enkooderin ulostulosta taas kuvan.

dec_input = tf.keras.Input(shape=(64,))
dec_dense1 = tf.keras.layers.Dense(7*7*64,activation='relu')(dec_input)
dec_reshape = tf.keras.layers.Reshape(target_shape=(7,7,64))(dec_dense1)
dec_convt1 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=(2,2),padding='same',activation='relu')(dec_reshape)
dec_convt2 = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=(2,2),padding='same',activation='relu')(dec_convt1)
dec_out = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,strides=(1,1),padding='same',activation="sigmoid")(dec_convt2)
decoder = tf.keras.Model(inputs=dec_input,
                         outputs=dec_out,
                         name='Decoder')
decoder.summary()

'''
Model: "Decoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_13 (InputLayer)        [(None, 64)]              0         
_________________________________________________________________
dense_19 (Dense)             (None, 3136)              203840    
_________________________________________________________________
reshape (Reshape)            (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 64)        36928     
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 32)        18464     
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         289       
=================================================================
Total params: 259,521
Trainable params: 259,521
Non-trainable params: 0
'''

#%%

# Summary - funktion tuloksesta nähdään,
# että Conv2D Transpose - kerrosten jälkeen ulostulona on 28x28 kokoinen mustavalkoinen kuva,
# joka on samassa muodossa kuin syötteenä tulevat kuvatkin.

# Yhdistetään luodut enkooderi ja dekooderi mallit.
#  Niin kuin notebookin alussa kuvassa nähtiin,
# enkooderin ulostulo luo pullonkaulaan vektorin,
# jonka avulla lähdetään luomaan kuvaa uudestaan dekooderilla.
# Voidaan ajatella, että enkooderi luo kuvasta "sisäisen käsityksen",
# ja dekooderin tehtävänä on oppia, miten tästä käsityksestä saa luotua samanlaisen kuvan,
# kuin tuli syötteenä autoenkooderille.

autoencoder_input = tf.keras.Input(shape=(28,28,1),name='Autoencoder_input')
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = tf.keras.Model(inputs=autoencoder_input, 
                             outputs=decoder_out,
                             name='Autoencoder')
autoencoder.summary()

'''
Model: "Autoencoder"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Autoencoder_input (InputLaye [(None, 28, 28, 1)]       0         
_________________________________________________________________
Encoder (Functional)         (None, 64)                166336    
_________________________________________________________________
Decoder (Functional)         (None, 28, 28, 1)         259521    
=================================================================
Total params: 425,857
Trainable params: 425,857
Non-trainable params: 0
'''

#%%

# Nyt koko autoenkooderin sisääntulo menee ensin enkooderille sisääntuloksi
# ja enkooderin ulostulo menee dekooderille sisääntuloksi.

# Kun autoenkooderi koulutetaan, sille annetaan ominaisuuksina ja ennustettavana arvoina samat kuvat.
# Aikaisemmin käytetyt 'train_X' ja 'train_y' muuttujat ovat siis molemmat samoja MNIST - datasetin kuvia.

autoencoder.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001),
                    loss = 'mean_squared_error')
autoencoder.fit(images,images,epochs=5,batch_size=16)

'''
Epoch 1/5
3750/3750 [==============================] - 122s 33ms/step - loss: 0.0258
Epoch 2/5
3750/3750 [==============================] - 126s 34ms/step - loss: 0.0061
Epoch 3/5
3750/3750 [==============================] - 126s 33ms/step - loss: 0.0042
Epoch 4/5
3750/3750 [==============================] - 124s 33ms/step - loss: 0.0034
Epoch 5/5
3750/3750 [==============================] - 128s 34ms/step - loss: 0.0030
'''

#%%
# TESTIKUVA 1

# Piirretään testikuva
import matplotlib.pyplot as plt
test_img = images[0]
test_img = test_img.reshape(test_img.shape[0],test_img.shape[1])
plt.imshow(test_img)
plt.show()

# Luodaan aikaisemmasta testikuvasta autoenkooderilla kuva uudelleen.
test_img = test_img.reshape(1,28,28,1)
prediction = autoencoder.predict(test_img)
prediction = prediction.reshape(prediction.shape[1],prediction.shape[2])
plt.imshow(prediction)
plt.show()

#%%
# TESTIKUVA 2

# Piirretään testikuva
import matplotlib.pyplot as plt
test_img = images[34]
test_img = test_img.reshape(test_img.shape[0],test_img.shape[1])
plt.imshow(test_img)
plt.show()

# Luodaan aikaisemmasta testikuvasta autoenkooderilla kuva uudelleen.
test_img = test_img.reshape(1,28,28,1)
prediction = autoencoder.predict(test_img)
prediction = prediction.reshape(prediction.shape[1],prediction.shape[2])
plt.imshow(prediction)
plt.show()

#%%
# TESTIKUVA x n
start = 11
count = 10
for i in range(start, start+count):    
    # Piirretään testikuva
    import matplotlib.pyplot as plt
    test_img = images[i]
    test_img = test_img.reshape(test_img.shape[0],test_img.shape[1])
    plt.imshow(test_img)
    plt.show()
    
    # Luodaan aikaisemmasta testikuvasta autoenkooderilla kuva uudelleen.
    test_img = test_img.reshape(1,28,28,1)
    prediction = autoencoder.predict(test_img)
    prediction = prediction.reshape(prediction.shape[1],prediction.shape[2])
    plt.imshow(prediction)
    plt.show()

#%%

# Huomataan, että autoencoder - malli loi kuvassa olevan numeron melkein samanlaiseksi, mutta ei täysin yksi yhteen.
# Mitä pienempi "latent" avaruudessa oleva vektori on, sitä vähemmän tietoa dekooderilla on,
# millä piirtää numeroita takaisin samanlaiseksi.

# Autoenkoodereita käytetään kuvien käsittelyssä:

# Kuvien siistiminen, kohinan poisto
# Kuvien kompressointi
# Super-resolution eli kuvan resoluution kasvattaminen

#%%

# Generative Adversarial Network (GAN)
# GAN on neuroverkkomalli, joka koostuu kahdesta erillisestä neuroverkosta.
# Ensimmäisen neuroverkon (generaattorin) tehtävänä on luoda esim. aidolta näyttäviä kuvia kissoista.
# Toisen neuroverkon (diskriminaattori) tehtävänä on päätellä, onko kuva oikea vai väärennös.

# Voidaan ajatella, että generaattori on rikollinen,
# joka yrittää luoda väärennettyjä seteleitä ja diskriminaattori on etsivä,
# joka yrittää tunnistaa väärennetyt setelit.
# Koulutuksen aikana ideaalissa tapauksessa molemmat verkot kehittyvät,
# eli generaattori luo paremman näköisiä väärennöksiä ja diskriminaattori oppii tunnistamaan väärennökset paremmin.

# GAN verkkojen osalta on luotu monia mielenkiintoisia käyttötapauksia, kuten:

# Nvidian StyleGAN, joka luo aitojen näköisiä kuvia ihmisten naamoista. Sivulla https://www.thispersondoesnotexist.com/ voi nähdä kyseisen neuroverkon tuottamia kuvia
# Nvidian GauGAN, joka luo piirroksista aitojen näköisiä maisemia. Kokeile vaikka itse selaimessa tästä.
# Deepfake, jossa kuvassa tai videossa esiintyvän henkilön naama korvataan toisen ihmisen naamalla. Valmista Python - ohjelmaa voi kokeilla esim tästä Github repositoriosta.
# GAN - verkkoja voi luoda myös Kerasissa. Prosessi vie kuitenkin paljon laskentatehoa ja on altis epäonnistumaan herkästi, joten GAN - verkkojen luontia Kerassilla ei käydä tässä materiaalissa läpi.