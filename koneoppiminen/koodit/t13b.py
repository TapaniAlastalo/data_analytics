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

print("train")
train_it = tf.keras.preprocessing.image.DirectoryIterator(
    path,
    imageDataGenerator,
    batch_size=64,
    color_mode="rgb",
    #classes=imgClasses,
    classes=None, 
    class_mode='binary', #class_mode='categorical',
    target_size=(256, 256),
    save_format='jpg',
    shuffle=True,
    seed=42
    )
print(train_it)

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

print("validate")
val_it = tf.keras.preprocessing.image.DirectoryIterator(
    path,
    imageDataGenerator,
    batch_size=32,
    color_mode="rgb",
    #classes=imgClasses,
    classes=None, 
    class_mode='binary', #class_mode='categorical',
    target_size=(256, 256),
    save_format='jpg',
    shuffle=True,
    seed=43
    )
print(val_it)


print("test")
test_it = tf.keras.preprocessing.image.DirectoryIterator(
    path,
    imageDataGenerator,
    batch_size=32,
    color_mode="rgb",
    #classes=imgClasses,
    classes=None, 
    class_mode='binary', #class_mode='categorical',
    target_size=(256, 256),
    save_format='jpg',
    shuffle=True,
    seed=44
    )
print(test_it)


#model
print('model')
#model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#model.add(Activation(‘relu’))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation(‘relu’))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation(‘relu’))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation(‘relu’))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation(‘sigmoid’))
#model.compile(loss=’binary_crossentropy’,
#optimizer=’rmsprop’,
#metrics=[‘accuracy’])
###
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),    
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')    
    ])

model.compile(loss='binary_crossentropy',  # loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['categorical_accuracy'])

#model.fit(x_train_flat, y_train, validation_data=(x_test_flat, y_test), epochs=10, batch_size=100)
#ennuste_test = model.predict(x_test_flat)

###
epochs = 20
history = model.fit_generator(train_it, steps_per_epoch=16, epochs=epochs, validation_data=val_it, validation_steps=8)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)



# evaluate model
#loss = model.evaluate_generator(test_it, steps=24)

# make a prediction
#yhat = model.predict_generator(predict_it, steps=24)
