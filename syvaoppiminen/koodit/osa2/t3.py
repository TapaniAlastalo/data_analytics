import tensorflow as tf
import os
# Graphviz exe:n polun määritys
os.environ['PATH'] = os.environ['PATH']+';'+ os.environ['CONDA_PREFIX'] + r"\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz"

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

# Neuroverkon rakenne kuvaksi
tf.keras.utils.plot_model(model_cnn,dpi=70,show_shapes=True)

#%% 
# Lataa malliin painoarvot "weights.h5" tiedostosta.
model_cnn.load_weights('weights.h5')
#model_cnn.layers[0].get_weights() 

#%%
# Aja tehtävän viimeinen "Vastaukset" solu
model_cnn.summary()