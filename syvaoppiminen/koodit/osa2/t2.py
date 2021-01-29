# Tehtävän toteutus

from IPython.display import Image
import tensorflow as tf


# Lataa MNIST datasetti.
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()

# kutistetaan 10-osaan
train_slice = 6000
test_slice = 1000
train_X = train_X[:train_slice,:]
train_y = train_y[:train_slice]
test_X = test_X[:test_slice,:]
test_y = test_y[:test_slice]

# lisätään väriavaruus 1 - mustavalko
train_X = train_X.reshape((train_X.shape[0],28,28,1))

print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)
# (60000, 28, 28, 1)


#%%
# Luo konvoluutioneuroverkko, joka vie klassifiointikerroksille 32 feature - matriisia, joiden korkeus ja leveys on 6.
layer_input = tf.keras.Input(shape=(28,28,1)) 
#model_input = tf.keras.Input(shape=(train_X.shape[1],train_X.shape[2],1)) 
# konvoluutiokerros
layer_conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=1, padding='same')(layer_input)
layer_maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(layer_conv1)
layer_maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(layer_maxpool1)
layer_conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=1, padding='valid')(layer_maxpool2)

print(layer_input)
print(layer_conv2)

#%%
layer_flatten = tf.keras.layers.Flatten()(layer_conv2)
layer_output = tf.keras.layers.Dense(1, activation='softmax')(layer_flatten)

# model_output = tf.keras.layers.Conv2D(filters=32, kernel_size=(2,2),strides=1, padding='valid')(model_maxpool2)
model_mnist = tf.keras.Model(inputs=layer_input,
                        outputs=layer_output)

# Todista neuroverkon toimivuus kouluttamalla MNIST datasettiä muutama epookki.
model_mnist.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['categorical_accuracy'])

model_mnist.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=4, batch_size=100)
ennuste = model_mnist.predict(test_X)
#print(ennuste)


#%%
# Aja "Tehtävän vastaukset" solu

# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. 

# Sijoita layer_output muuttujaan sen konvoluutikerroksen ulostulon muoto, missä kerroksesta tulee ulos 32 feature matriisia, joiden korkeus ja leveys on 6
# Esim. layer_output = model.layers[5].output.shape
print(layer_output)
# Sijoita luomasi malli model_mnist - muuttujaan.
model_mnist.summary()