import tensorflow as tf
from IPython.display import Image
# Jos käytät Windowsia ja Anaconda - paketinhallintaa:
# Voit korjata "Graphviz executable not found" virheen määrittelemällä Graphviz exe - tiedostojen sijainnit käsin alla olevalla komennolla
import os
#os.environ['PATH'] = os.environ['PATH']+';'+ os.environ['CONDA_PREFIX'] + r"\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz"

try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot1
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot1
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot1
    except ImportError:
      pydot = None



#%%

    
# Model 1; sisääntulo, kaksi piilotettua ja ulostulokerros 2:lle
model1_input = tf.keras.Input(shape=(2,))
model1_dense1 = tf.keras.layers.Dense(10,activation='relu')(model1_input)
model1_dense2 = tf.keras.layers.Dense(10,activation='relu')(model1_dense1)
model1_output = tf.keras.layers.Dense(2,activation='linear')(model1_dense2)
    
# Model 2
model2_input = tf.keras.Input(shape=(2,))
model2_concat = tf.keras.layers.concatenate([model2_input, model1_output])
model2_dense1 = tf.keras.layers.Dense(10,activation='relu')(model2_concat)
model2_dropout = tf.keras.layers.Dropout(0.2, input_shape=(10,))(model2_dense1)
model2_output = tf.keras.layers.Dense(2,activation='relu')(model2_dropout)

# Model 3
model3_input =tf.keras.Input(shape=(2,))
model3_concat = tf.keras.layers.concatenate([model3_input, model1_output])
model3_dense1 = tf.keras.layers.Dense(10,activation='relu')(model3_concat)
model3_dropout = tf.keras.layers.Dropout(0.2, input_shape=(10,))(model3_dense1)
model3_output = tf.keras.layers.Dense(2,activation='relu')(model3_dropout)

# Yhdistetään Model2 ja Model3 ulostulot
model4_concat = tf.keras.layers.add([model2_output, model3_output])
model4_dense1 = tf.keras.layers.Dense(4,activation='relu')(model4_concat)
model4_output = tf.keras.layers.Dense(1,activation='sigmoid')(model4_dense1)

# Määritellään sisääntulot ja ulostulot
model_full = tf.keras.Model(inputs=[model1_input, model2_input, model3_input],
                               outputs=model4_output)

# Piirretään lopputuotos
tf.keras.utils.plot_model(model_full, dpi=70, show_shapes=True)

# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. Sijoita luomasi malli model_functional - muuttujaan.
model_full.summary()