#mport pandas as pd
import tensorflow as tf
import numpy as np

# Use GPU
tf.test.is_gpu_available(cuda_only=True)

# Lataa clickbait ja ei-clickbait otsikot clickbait_data.txt ja non_clickbait_data.txt tiedostoista data - kansiosta
file1 = open('c:/data/clickbait_data.txt', 'r', encoding='UTF-8')
clickbait_lines = file1.readlines()
np1 = np.array(clickbait_lines)
print(np1.shape)

file2 = open('c:/data/non_clickbait_data.txt', 'r', encoding='UTF-8')
non_clickbait_lines = file2.readlines()
np2 = np.array(non_clickbait_lines)
print(np2.shape)

# muutetaan listat numpy - taulukoiksi
features = np.append(np1, np2, axis=0)
# Leimaa otsikot 0 tai 1 luokkaan (clickbait vai ei)
labels = np.append(np.ones((np1.size)), np.zeros((np2.size)), axis=0)   

#%%
# Jaa data koulutus- ja testidataan (80% koulutusdataa 20% testidataa jako)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(features,labels, test_size=0.2)

# Tokenizer.
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(features)

# tekstit sekvenssiksi
train_X = tokenizer.texts_to_sequences(train_X)
test_X = tokenizer.texts_to_sequences(test_X)

# sekvenssit samankokoiseksi
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post', maxlen=256)
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post', maxlen=256)
print(train_X.shape)
print(test_X.shape)

'''
(46400, 256)
(11601, 256)
'''
# Koulutusdatassa on 46400 kappaletta arvosteluita, joiden pituus on 256 sanaa.
# Koulutusdatassa on 25000 kappaletta arvosteluita, joiden pituus on 256 sanaa.

#%%
# Nyt voimme syöttää enkoodatun tekstin neuroverkkomallille.
# Luodaan RNN - malli, jossa käytämme aikaisemmin mainittua Embedding - kerrosta.
# Embedding - kerroksen sanavektorit syötetään LSTM - kerrokselle,
# jonka tarkoituksena on oppia sanojen merkitys lauseessa.

input_layer = tf.keras.Input(shape=(256,))
emb = tf.keras.layers.Embedding(20000,2,input_length=train_X.shape[1])(input_layer)
lstm = tf.keras.layers.LSTM(64,return_sequences=True,dropout=0.1)(emb)
do = tf.keras.layers.Dropout(0.5)(lstm)
flat = tf.keras.layers.Flatten()(do)
dense = tf.keras.layers.Dense(64,activation='relu')(flat)
out = tf.keras.layers.Dense(1,activation='sigmoid')(dense)
model = tf.keras.Model(inputs=input_layer,
                    outputs=out)

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.fit(train_X,
          train_y,
          validation_data=(test_X,test_y),
          epochs=4)

'''
runcell(6, 'R:/Koodaus/repos/jamk/data_analytics/syvaoppiminen/koodit/osa3/nlp.py')
Epoch 1/10
782/782 [==============================] - 153s 196ms/step - loss: 0.5860 - accuracy: 0.6655 - val_loss: 0.4478 - val_accuracy: 0.7921
Epoch 2/10
782/782 [==============================] - 157s 201ms/step - loss: 0.3758 - accuracy: 0.8341 - val_loss: 0.3657 - val_accuracy: 0.8384
Epoch 3/10
782/782 [==============================] - 162s 207ms/step - loss: 0.3033 - accuracy: 0.8739 - val_loss: 0.3589 - val_accuracy: 0.8476
Epoch 4/10
782/782 [==============================] - 162s 207ms/step - loss: 0.2608 - accuracy: 0.8944 - val_loss: 0.3503 - val_accuracy: 0.8458
Epoch 5/10
782/782 [==============================] - 159s 204ms/step - loss: 0.2259 - accuracy: 0.9109 - val_loss: 0.3147 - val_accuracy: 0.8696
Epoch 6/10
782/782 [==============================] - 157s 201ms/step - loss: 0.1981 - accuracy: 0.9241 - val_loss: 0.3167 - val_accuracy: 0.8651
Epoch 7/10
782/782 [==============================] - 158s 202ms/step - loss: 0.1755 - accuracy: 0.9350 - val_loss: 0.3309 - val_accuracy: 0.8690
Epoch 8/10
782/782 [==============================] - 160s 205ms/step - loss: 0.1566 - accuracy: 0.9428 - val_loss: 0.3323 - val_accuracy: 0.8688
Epoch 9/10
782/782 [==============================] - 158s 201ms/step - loss: 0.1423 - accuracy: 0.9486 - val_loss: 0.3485 - val_accuracy: 0.8668
Epoch 10/10
782/782 [==============================] - 157s 201ms/step - loss: 0.1275 - accuracy: 0.9543 - val_loss: 0.3724 - val_accuracy: 0.8668
'''

#%%
# Visualisoidaan Embedding - kerros luomalla osamalli, jossa on koko mallin ja embedding - kerroksen ulostulo.
import matplotlib.pyplot as plt
submodel = tf.keras.Model(inputs=model.layers[0].input,
                          outputs=model.layers[1].output)
test_words = ["awesome shit hilarious oscar excellent bad unique terrible vile crap stupid magnificent stupendous terrific astounding exquisite superb poor ugly repulsive crude second-rate ok adequate tolerable decent splendid"]
test_words_seq = tokenizer.texts_to_sequences(test_words)
test_words_seq = tf.keras.preprocessing.sequence.pad_sequences(test_words_seq, padding='post', maxlen=256)
embedding_out = submodel(test_words_seq).numpy()
embedding_out = embedding_out[0]
plt.figure(figsize=(16,12))
plt.scatter(embedding_out[:,0], embedding_out[:,1])
review_words = test_words[0].split(' ')
for i, txt in enumerate(review_words):
    plt.annotate(txt, (embedding_out[i,0], embedding_out[i,1]))
plt.show()

# Kuvaajasta nähdään, kuinka Embedding - kerros erottaa neutraalit, negatiiviset ja positiiviset adjektiivit toisistaan.

