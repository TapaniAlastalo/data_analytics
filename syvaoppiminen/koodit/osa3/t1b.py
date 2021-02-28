# Tehtävän toteutus

import numpy as np

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
print(features[0], labels[0])
print(features.shape, labels.shape)
    
# muutetaan listat numpy - taulukoiksi
features, labels = np.array(features), np.array(labels)
print(features[0], labels[0])
print(features.shape,labels.shape)

#%%
# Jaa data koulutus- ja testidataan (80% koulutusdataa 20% testidataa jako)
divider = int(len(features) / 1.25)
train_X, train_y = features[:len(features) // divider], labels[:len(features) // divider]
test_X, test_y = features[len(features) // divider:], labels[len(features) // divider:]

print("shape")
print(train_X.shape,train_y.shape)
print(test_X.shape,test_y.shape)


#%%

import tensorflow as tf

word_to_index = tf.keras.datasets.imdb.get_word_index()
# Luodaan sanakirja. Sanakirjassa sanan perusteella löytää sitä vastaavan indeksin
# Ensimmäiset 4 indeksiä on varattu sanakirjassa esim. aloitus - ja tuntemattomille merkeille.
word_to_index = {word:(index+3) for word,index in word_to_index.items()} 
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1 # Arvostelun alku
word_to_index["<UNK>"] = 2  # Tuntematon sana, eli sana on jäänyt sanakirjan ulkopuolelle
word_to_index["<UNUSED>"] = 3

# Käännetään aikaisempi sanakirja niin, että indeksin perusteella löytää sanan. Tämän avulla voidaan kääntää indeksejä sisältävät sekvenssit lauseiksi.
index_to_word = {index:word for word,index in word_to_index.items()}

def review_to_text(review):
    # Etsitään indeksistä numeroa vastaava sana, lisätään se listaan ja luodaan listasta merkkijono
    words_list = [index_to_word.get(x,'?') for x in review]
    words_str = ' '.join(words_list)
    return words_str
# Muutetaan koulutus- ja testidata tekstiksi
train_reviews = list(map(review_to_text,train_X))
test_reviews = list(map(review_to_text,test_X))
#train_reviews[1]

'''

# sisääntulo pitää muotoilla 3D muotoon eli (samples, time_steps, features)
# Tässä tapauksissa ominaisuuksia on vain yksi kappale ja time_steps on 10 eli kymmenen viimeisintä arvoa.

train_X = train_X.reshape(train_X.shape[0], 1, 1)
test_X =  test_X.reshape(test_X.shape[0], 1, 1)

#train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
#test_X =  test_X.reshape(test_X.shape[0], test_X.shape[1], 1)


print(train_X.shape,train_y.shape)
print(test_X.shape,test_y.shape)
'''
#%% 1

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token='<UNK>')
tokenizer.fit_on_texts(train_reviews)

# tekstit sekvenssiksi
train_X = tokenizer.texts_to_sequences(train_reviews)
test_X = tokenizer.texts_to_sequences(test_reviews)

# sekvenssit samankokoiseksi
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post', maxlen=256)
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post', maxlen=256)
train_X.shape

#%%
# Luo RNN - malli, joka ennustaa, onko otsikko clickbait vai ei
print("model")
'''
input_layer = tf.keras.Input(shape=(256,))
emb = tf.keras.layers.Embedding(20000,2,input_length=train_X.shape[1])(input_layer)
lstm = tf.keras.layers.LSTM(64,return_sequences=True,dropout=0.1)(emb)
do = tf.keras.layers.Dropout(0.5)(lstm)
flat = tf.keras.layers.Flatten()(do)
dense = tf.keras.layers.Dense(64,activation='relu')(flat)
out = tf.keras.layers.Dense(1,activation='sigmoid')(dense)
model = tf.keras.Model(inputs=input_layer,
                    outputs=out)
'''
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])


#%%
# Tulosta mallin tarkkuus evaluate - funktiolla
# Seuraavaksi luomme neuroverkon.
# Vertailun vuoksi luodaan ensin täysin yhdistetty neuroverkko, jossa on kaksi piiloitettua kerrosta.
print("compile")
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.fit(train_X, train_y, validation_data=(test_X,test_y), validation_setps=30, epochs=3)

results = model.evaluate(test_X, test_y, verbose=0)
print("predict")
predictions = model.predict(test_X)
print(f"Test loss {results[0]}")

#%%
# Aja "Tehtävän vastaukset" solu

# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. Sijoita results - muuttujaan funktion model.evaluate() tulos.
# Muista määrittää model.compile() - funktioon seurattavaksi suureeksi metrics=['accuracy'], jotta näät, kuinka suuri osa neuroverkon ennustuksista on oikein.
print(f"Test Loss:{results[0]} Test Accuracy:{results[1]*100}%")