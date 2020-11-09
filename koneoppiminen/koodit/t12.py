import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.imshow(x_train[0], cmap='Greys')

# tehdään x_train taulukosta 1-ulotteinen, säilytetään koko ja määritellään mustavalkoiseksi (1)
x_train_flat = x_train.reshape(60000, 28, 28, 1)
x_test_flat = x_test.reshape(10000, 28, 28, 1)
x_train_flat = x_train_flat/255
x_test_flat = x_test_flat/255

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(30, kernel_size=5, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(15, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),    
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')    
    ])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['categorical_accuracy'])

model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=10, batch_size=100)
ennuste_test = model.predict(x_test_flat)

plt.imshow(x_test[268], cmap='Greys')
plt.imshow(x_test[9749], cmap='Greys')
plt.imshow(x_test[9768], cmap='Greys')
plt.imshow(x_test[9982], cmap='Greys')

# uudelleen opetus
model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=10, batch_size=100)
# mallin tallennus tiedostoon
model.save('mnistconvmodel.h12')
