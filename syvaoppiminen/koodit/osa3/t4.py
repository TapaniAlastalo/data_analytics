# Ladataan tarvittavat kirjastot
import tensorflow as tf
import numpy as np
# Puhdas MNIST datasetti, josta otetaan kymmenesosa kuvista datasettiin
(train_X, _), (test_X, _) = tf.keras.datasets.mnist.load_data()
train_X, test_X = train_X[0:len(train_X) // 10], test_X[0:len(test_X) // 10]
train_X, test_X = train_X / 255.0, test_X / 255.0
train_X, test_X = train_X.reshape((train_X.shape[0],28,28,1)), test_X.reshape((test_X.shape[0],28,28,1))

# Luodaan suttuisia kuvia MNIST datasetistä laittamalla kuviin kohinaa
noise = np.random.normal(loc=0.5, scale=0.75, size=train_X.shape)
train_X_noisy = train_X + noise
noise = np.random.normal(loc=0.5, scale=0.75, size=test_X.shape)
test_X_noisy = test_X + noise
train_X_noisy = np.clip(train_X_noisy, 0., 1.) 
test_X_noisy = np.clip(test_X_noisy, 0., 1.)

print(train_X.shape)
print(train_X_noisy.shape)
print(test_X_noisy.shape)

#%%
# Tehtävän toteutus

#%%
# Luodaan enkooderi
enc_input = tf.keras.Input(shape=(28,28,1))
enc_conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=(2,2),activation='relu')(enc_input)
enc_conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=(2,2),activation='relu')(enc_conv1)
# luodaan tiivistetty luonnos kuvasta Flatten() ja Dense kerroksella,
enc_flatten = tf.keras.layers.Flatten()(enc_conv2)
# Viimeisessä Dense kerroksessa luodaan ennuste kuvasta, laitetaan aktivaatiofunktioksi 'linear'
enc_out = tf.keras.layers.Dense(64,activation='linear')(enc_flatten)
encoder = tf.keras.Model(inputs=enc_input,
                         outputs=enc_out,
                         name='Encoder')
encoder.summary()

#%%
# Luodaan dekooderi. Dekooderin tehtävä on luoda tiivistetty enkooderin tieto takaisin kuvaksi.
dec_input = tf.keras.Input(shape=(64,))
dec_dense1 = tf.keras.layers.Dense(7*7*64,activation='relu')(dec_input)
dec_reshape = tf.keras.layers.Reshape(target_shape=(7,7,64))(dec_dense1)
# Conv2DTranspose kerroksella pienempi sisääntulo muutetaan suuremmaksi sisääntuloksi (dekonvoluutio)
dec_convt1 = tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=(2,2),padding='same',activation='relu')(dec_reshape)
dec_convt2 = tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=(2,2),padding='same',activation='relu')(dec_convt1)
dec_out = tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,strides=(1,1),padding='same',activation="sigmoid")(dec_convt2)
decoder = tf.keras.Model(inputs=dec_input,
                         outputs=dec_out,
                         name='Decoder')
decoder.summary()

#%%
# Yhdistetään luodut enkooderi ja dekooderi mallit autoenkooderilla. 
autoencoder_input = tf.keras.Input(shape=(28,28,1),name='Autoencoder_input')
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = tf.keras.Model(inputs=autoencoder_input, 
                             outputs=decoder_out,
                             name='Autoencoder')
autoencoder.summary()

#%%

# Autoenkooderille annetaan ominaisuuksina train_X_noisy ja ennustettavana train_X.
autoencoder.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001),
                    loss = 'mean_squared_error')
autoencoder.fit(train_X_noisy,train_X, epochs=5, batch_size=16)

#%%
# Todista autoenkooderin toiminta:
# Piirrä Matplotlibillä testidatasetistä suttuinen kuva (esim. test_X_noisy[10])
# Syötä suttuinen kuva autoenkooderille.
# Piirrä autoenkooderin ulostulona antama kuva.

# Piirretään testikuva
import matplotlib.pyplot as plt
test_img = test_X_noisy[10]
test_img = test_img.reshape(test_img.shape[0],test_img.shape[1])
plt.imshow(test_img)
plt.show()

# Luodaan aikaisemmasta testikuvasta autoenkooderilla kuva uudelleen.
test_img = test_img.reshape(1,28,28,1)
prediction = autoencoder.predict(test_img)
prediction = prediction.reshape(prediction.shape[1],prediction.shape[2])
plt.imshow(prediction)
plt.show()

cleaned_img = prediction

# Aja "Tehtävän vastaukset" solu

#%%
# Tehtävän vastaukset.  Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. 
# Syötä cleaned_img - muuttujaan autoenkooderin ulostulo. Muista muokata sitä ennen ulostulo takaisin 28x28 matriisiksi, jotta sen voi syöttää plt.imshow() funktiolle.
plt.imshow(cleaned_img)