import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import cv2

#from tensorflow import keras
# example of converting an image with the Keras API
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
#from tensorflow.keras.preprocessing import image_dataset_from_directory

#im = cv2.imread('data/train/train/cat.0.jpg')
#img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB
#cv2.imwrite('cat.png', img) 
#print (type(img))

# load the image
img = load_img('data/train/cats/cat.0.jpg')
print("Orignal:" ,type(img))

# convert to numpy array
img_array = img_to_array(img)
print("NumPy array info:") 
print(type(img_array))    

print("type:",img_array.dtype)
print("shape:",img_array.shape)

print("test")

path = 'data/train'
imageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator()
imgClasses =  ["dogs", "cats"]

#(x_train, y_train), (x_test, y_test) 

loaded = tf.keras.preprocessing.image.DirectoryIterator(
    path,
    imageDataGenerator,
    color_mode="rgb",
    #classes=imgClasses,
    classes=None, class_mode='categorical',
    target_size=(256, 256),
    save_format='jpg'
    )

print("loaded")
print(loaded)

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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
