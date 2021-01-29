import tensorflow as tf

# Lataa Fashion MNIST datasetti ajamalla tehtävän ensimmäinen solu.
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.fashion_mnist.load_data()

#%%
# Luo alla olevan kuvan mukainen neuroverkkomalli.
model_cnn = tf.keras.Sequential([    
    tf.keras.layers.InputLayer((28,28,1)),
    #tf.keras.layers.Dense(1, input_shape=(28,28,1), activation='relu'), # Tarvittava neuronien määrä nähdään kuvasta "output" - kentästä
    #tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(24, kernel_size=(5,5), activation='relu', strides=1, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(48, kernel_size=(5,5), activation='relu', strides=1, padding='same'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu', strides=1, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)
])
# Lataa malliin painoarvot "weights.h5" tiedostosta.
model_cnn.load_weights('weights.h5')
# Tehtävän toteutus tähän
#%%
#train_X = train_X.reshape((train_X.shape[0],28,28,1))
#test_X = test_X.reshape((train_X.shape[0],28,28,1))
print(train_X.shape)
# Poista äskeisessä tehtävässä luodusta mallista klassifikaatiokerrokset. (eli Flatten ja kaikki sen jälkeiset kerrokset)

model_cnn2 = tf.keras.Model(inputs = model_cnn.input,
                             outputs = model_cnn.output)
#model_cnn2 = model_cnn
model_cnn2 = model_cnn2.Layer.call(input_shape=(32,32,3), include_top=False)
#model_cnn2.summary()


# Jäädytä loput kerrokset.
for layer in model_cnn2.layers:
    layer.trainable = False
    #model_cnn2.layers.remove(layer)    
    #layer.pop()
    
model_cnn2.summary()

# Luo klassifikaatiokerrokset ja lisää ne malliin.
flatten = tf.keras.layers.Flatten()(model_cnn2.output)
new_dense1 = tf.keras.layers.Dense(256,activation='relu')(flatten)
new_output = tf.keras.layers.Dense(10,activation='softmax')(new_dense1)
# tehdään uusi malli olio
model_cnn3 = tf.keras.Model(inputs = model_cnn2.input,
                            outputs = model_cnn2.output)
                             #outputs = new_output)
#model_cnn3.summary()


#%%
# Kouluta mallia Fashion MNIST datasetillä muutama kierros (epoch) käyttäen train_X ja train_y koulutusdataa.
#model_cnn3.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])
#model_cnn3.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['mean_squared_error'])
model_cnn3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model_cnn3.fit(train_X, train_y, epochs=4, batch_size=5)
#model_cnn3.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=4, batch_size=2)


#model_cnn3.fit(test_X, test_y,epochs=5,verbose=0)
results = model_cnn2.predict(test_X)

# Aja tehtävän viimeinen "Vastaukset" solu.
#%%
# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. Sijoita results - muuttujaan funktion model.evaluate() tulos.
# Muista määrittää model.compile() - funktioon seurattavaksi suureeksi metrics=['accuracy'], jotta näät, kuinka suuri osa neuroverkon ennustuksista on oikein.
print(f"Test Loss:{results[0]} Test Accuracy:{results[1]*100}%")