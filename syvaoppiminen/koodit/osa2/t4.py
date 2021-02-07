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
model_cnn.load_weights('c:/data/weights.h5')

#%%
layer_input = tf.keras.Input(shape=(28,28,1)) 
#model_input = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2],1)) 
# konvoluutiokerros
layer_conv1 = tf.keras.layers.Conv2D(filters=24, kernel_size=(5,5),strides=1, padding='same', activation='relu')(layer_input)
layer_maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(layer_conv1)
layer_drop1 = tf.keras.layers.Dropout(0.2)(layer_maxpool1)
layer_conv2 = tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5),strides=1, padding='same', activation='relu')(layer_drop1)
layer_drop2 = tf.keras.layers.Dropout(0.4)(layer_conv2)
layer_conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5),strides=1, padding='same', activation='relu')(layer_drop2)
layer_maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(layer_conv3)
layer_flatten1 = tf.keras.layers.Flatten()(layer_maxpool2)
layer_dense1 = tf.keras.layers.Dense(256, activation='relu')(layer_flatten1)
layer_output = tf.keras.layers.Dense(10)(layer_dense1)

model_cnn_b = tf.keras.Model(inputs=layer_input, outputs=layer_output)
# Lataa malliin painoarvot "weights.h5" tiedostosta.
model_cnn_b.load_weights('c:/data/weights.h5')

# Tehtävän toteutus tähän
#%%
print(model_cnn)
print(model_cnn.output)
print(model_cnn.layers)
print(model_cnn.layers[6])

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

#%%
# Poista äskeisessä tehtävässä luodusta mallista klassifikaatiokerrokset. (eli Flatten ja kaikki sen jälkeiset kerrokset)
model_cnn2 = tf.keras.Model(inputs = model_cnn_b.input,
                             outputs = layer_maxpool2)
model_cnn2.summary()
model_cnn.summary()

#%%
#train_X = train_X.reshape((train_X.shape[0],28,28,1))
#test_X = test_X.reshape((train_X.shape[0],28,28,1))
print(train_X.shape)

#%%
#model_cnn2 = model_cnn
#model_cnn2 = model_cnn(input_shape=(32,32,3), include_top=False)
#model_cnn2.summary()


# Jäädytä loput kerrokset.
for layer in model_cnn2.layers:
    layer.trainable = False

# Luo klassifikaatiokerrokset ja lisää ne malliin.
new_flatten = tf.keras.layers.Flatten()(model_cnn2.output)
new_dense1 = tf.keras.layers.Dense(256,activation='relu')(new_flatten)
new_output = tf.keras.layers.Dense(10,activation='softmax')(new_dense1)
# tehdään uusi malli olio
model_cnn3 = tf.keras.Model(inputs = model_cnn2.input,
                            outputs = new_output)
                             
model_cnn3.summary()


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