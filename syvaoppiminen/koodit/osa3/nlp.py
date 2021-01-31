# Natural language Processing eli NLP on luonnollisen kielen käsittelyä neuroverkoilla.
# Tätä aihealuetta on kehitetty eteenpäin mm. Google Assistent ja Amazonin Alexan toimista.
# Yleiset käyttötapaukset ovat virtuaaliassistenttejä, jotka tunnistavat käyttäjän puheesta kysymyksen,
# käsittelevät kysymyksen ja etsivät avainsanojen perusteella vastauksen joko valmiista tietokannasta
# tai luomalla lauseen itse.

# Ongelma on, että miten teksti muokataan niin, että sen voi syöttää neuroverkoille.

# One-hot enkoodaus
# One-hot enkoodauksella tekstissä jokainen sana korvataan ns. one-hot vektorilla.
# Ongelma tässä on se, että jos datasetti, missä on 1000 eri sanaa muunnetaan one-hot enkoodauksella vektoreiksi,
# niin jokaisessa vektorissa olisi yksi arvo "päällä" ja 999 muuta arvoa "pois päältä".

# One-hot Kerassissa
# Luodaan esimerkkinä lista, jossa on satunnaisia lauseita
# ja enkoodataan nämä lauseet 'one_hot' - funktiolla Kerassissa.
# Vaikka funktion nimi on one_hot, niin se ei yleisen one_hot menetelmän mukaisesti luo vektoria,
# jossa olisi yksi arvo päällä, vaan kokonaislukuja, joista yksi kokonaisluku tarkoittaa yhtä sanaa.

import pandas as pd
import tensorflow as tf
import numpy as np
sentences = ['Kissa ei ole koira',
         'Koira ei ole kissa',
         'Musti on koira',
         'Mirri on kissa',
         'Tapani on töissä',
         'Annukka harrastaa jääkiekkoa',
         'Essi harrastaa jalkapalloa',
         'Matti pelaa biljardia']
# Käydään map - funktiolla listan lauseet läpi, syötetään lauseet one_hot - funktiolle. 
# One-hot funktion ensimmäinen parametri on lause, joka muutetaan numeeriseksi.
# Toinen parametri määrittää sen, kuinka monta eri sanaa one_hot funktio tunnistaa.
one_hot_sentences = list(map(lambda x: tf.keras.preprocessing.text.one_hot(x, 100),sentences))
print(one_hot_sentences)

'''
[[76, 13, 11, 72], [72, 13, 11, 76], [92, 47, 72], [64, 47, 76], [24, 47, 50], [73, 18, 70], [5, 18, 11], [60, 87, 29]]
'''
# Näämme, että lauseissa esiintyvät samat sanat on korvattu samoilla kokonaisluvuilla.

#%%

# Tokenizer
# Vaihtoehto one-hot enkoodaukselle on käyttää Kerasin Tokenizer - luokkaa.

# Fit_on_texts - funktiolla voi luoda tekstistä "sanakirjan", joka on järjestelty niin,
# että mitä pienempi sanan indeksiarvo on, sitä useammin sana esiintyy teksteissä.
# Tokenizer - luokka osaa myös poistaa välimerkit teksteistä automaattisesti.

# Kun sanakirja on luotu, texts-to-sequences - funktiolla voidaan muuttaa lauseet numeeriseen muotoon.
# Funktio korvaa sanakirjan avulla sanakirjasta löytyvän indeksin avulla.

# Esimerkkilauseita Aleksis Kiven Seitsemän veljestä - kirjasta
text = ['Sen läheisin ympäristö on kivinen tanner, mutta alempana alkaa pellot, joissa, ennenkuin talo oli häviöön mennyt, aaltoili teräinen vilja.',
        'Heidän isäänsä, joka oli ankaran innokas metsämies, kohtasi hänen parhaassa iässään äkisti surma, kun hän tappeli äkeän karhun kanssa.',
        'Pahoin oli mies haavoitettu, mutta pedonkin sekä kurkku että kylki nähtiin puukon viiltämänä ja hänen rintansa kiväärin tuiman luodin lävistämänä.',
        'He rakentelivat satimia, loukkuja, ansaita ja teerentarhoja surmaksi linnuille ja jäniksille.']

# Tokenizer luonti. Parametriin 'num_words' määritellään sanakirjan koko.
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=120)
# Sanakirjan luonti
tokenizer.fit_on_texts(text)

# Tekstit sekvensseiksi
text_seq = tokenizer.texts_to_sequences(text)
print(text_seq)

'''
[[5, 6, 7, 8, 9, 10, 3, 11, 12, 13, 14, 15, 16, 1, 17, 18, 19, 20, 21],
 [22, 23, 24, 1, 25, 26, 27, 28, 4, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
 [39, 1, 40, 41, 3, 42, 43, 44, 45, 46, 47, 48, 49, 2, 4, 50, 51, 52, 53, 54],
 [55, 56, 57, 58, 59, 2, 60, 61, 62, 2, 63]]
'''
# Sekvensseistä nähdään, että indeksit 1 ja 2 ovat 'oli' sekä 'ja' - sanoja,
# jotka esiintyvät teksteissä kolme kertaa. Sana 'mutta' esiintyy kaksi kertaa, ja sen indeksiksi on tullut 3.

#%%

# Kun tekstit on enkoodattu käyttäen one-hot tai tokenizer - menetelmää,
# tulee kaikki sekvenssit muuttaa samanpituisiksi.
# Voimme käyttää taas Kerassin valmista funktiota 'pad_sequences'.
# Parametri 'maxlen' on sekvenssin pituus, joka tulisi olla pisimmän enkoodatun lauseen pituus.
# Parametri 'padding' voi saada arvot 'pre' tai 'post', mikä määrää sen,
# täydennetäänkö sekvenssin alku vai loppu nollilla.

# Täytetään one-hot enkoodatut lauseet samanpituisiksi
padded_one_hot_sentences = tf.keras.preprocessing.sequence.pad_sequences(one_hot_sentences, maxlen=4, padding='post')
print(padded_one_hot_sentences)
'''
[[76 13 11 72]
 [72 13 11 76]
 [92 47 72  0]
 [64 47 76  0]
 [24 47 50  0]
 [73 18 70  0]
 [ 5 18 11  0]
 [60 87 29  0]]
'''

# Tokenizer - luokalla luodut sekvenssit samanpituisiksi
padded_text_seq = tf.keras.preprocessing.sequence.pad_sequences(text_seq, maxlen=25, padding='post')
print(padded_text_seq)
'''
[[ 5  6  7  8  9 10  3 11 12 13 14 15 16  1 17 18 19 20 21  0  0  0  0  0
   0]
 [22 23 24  1 25 26 27 28  4 29 30 31 32 33 34 35 36 37 38  0  0  0  0  0
   0]
 [39  1 40 41  3 42 43 44 45 46 47 48 49  2  4 50 51 52 53 54  0  0  0  0
   0]
 [55 56 57 58 59  2 60 61 62  2 63  0  0  0  0  0  0  0  0  0  0  0  0  0
   0]]
'''

#%%

# Word embedding
# Word embedding - menetelmällä pyritään antamaan samanlaisille sanoille samankaltaiset numeeriset esitykset.
# Esimerkiksi sanat kisa ja kissa ovat melkein identtiset, mutta merkitykset ovat kaukana toisistaan.
# Vastaavasti sanat kissa ja mirri eivät ole samannäköisiä, mutta ovat merkitykseltään samoja.
# Tavoituksena olisi siis luoda numeerinen esitys,
# jossa sanat kisa ja kissa ovat numeerisesti kaukana toisistaan ja kissa ja mirri taas lähellä toisiaan.

# Kerroksen sisääntulon muoto on yhtä suuri kuin sanakirjan koko.
# Embedding - kerroksen ulostulon koko saattaa vaihdella kadesta sarakkeesta 1024 sarakkeeseen
# riippuen sanakirjan koosta. Mitä enemmän sanoja ja dataa on, sitä enemmän voi oppia sanojen välisiä suhteita.

# Luodaan sisääntulokerros ja Embedding - kerros,
# ja luodaan ennuste one-hot enkoodatuille lauseille,
# jotta nähdään, minkälainen on Embedding - kerroksen ulostulo.

input_layer = tf.keras.Input(shape=(4,))
# sanakirjan koko on 100
embedding_layer = tf.keras.layers.Embedding(100,2,input_length=4)(input_layer)
model = tf.keras.Model(inputs=input_layer,
             outputs=embedding_layer)
model.summary()
'''

Model: "functional_19"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         [(None, 4)]               0         
_________________________________________________________________
embedding (Embedding)        (None, 4, 2)              200       
=================================================================
Total params: 200
Trainable params: 200
Non-trainable params: 0
_________________________________________________________________
'''

preds = model.predict(padded_one_hot_sentences)
print(preds)
'''
[[[ 1.79501499e-02 -4.98035399e-02]
  [-3.82237165e-02 -3.69405447e-05]
  [ 8.31284487e-03  1.70936545e-02]
  [-4.32756043e-02 -3.67032099e-02]]

 [[-4.32756043e-02 -3.67032099e-02]
  [-3.82237165e-02 -3.69405447e-05]
  [ 8.31284487e-03  1.70936545e-02]
  [ 1.79501499e-02 -4.98035399e-02]]

 [[ 4.10123032e-02 -5.30551178e-04]
  [-2.04693431e-02 -1.22905891e-03]
  [-4.32756043e-02 -3.67032099e-02]
  [-3.50114389e-02  4.64541872e-02]]

 [[ 3.21174207e-02  4.73175963e-02]
  [-2.04693431e-02 -1.22905891e-03]
  [ 1.79501499e-02 -4.98035399e-02]
  [-3.50114389e-02  4.64541872e-02]]

 [[-4.17037388e-02  1.67252790e-02]
  [-2.04693431e-02 -1.22905891e-03]
  [-3.03383458e-02  7.36605592e-03]
  [-3.50114389e-02  4.64541872e-02]]

 [[ 2.08624371e-02  1.90880917e-02]
  [-4.30219960e-02  3.60267145e-02]
  [-2.60602510e-02 -2.93437548e-02]
  [-3.50114389e-02  4.64541872e-02]]

 [[ 1.49637643e-02  4.30670522e-02]
  [-4.30219960e-02  3.60267145e-02]
  [ 8.31284487e-03  1.70936545e-02]
  [-3.50114389e-02  4.64541872e-02]]

 [[ 4.40347651e-02  3.40666529e-03]
  [ 7.08821760e-03  9.23167475e-03]
  [-1.97124719e-02  4.50992286e-02]
  [-3.50114389e-02  4.64541872e-02]]]
'''
# Embedding - kerrokseen tuli syötteenä kahdeksan lausetta, joista jokainen oli täydennetty neljän pituisiksi.
# Ulostulon dimensioiksi määritettiin kaksi, joten Embedding - kerroksen ulostulo on muotoa:

print(preds.shape)
'''
(8, 4, 2)
'''

#%%

# Tekstin klassifiointi
# Tekstin klassifiointi on yleinen esimerkki NLP - ongelmasta.
# Yksi yleinen ongelma on lauseiden luokittelu eri kategorioihin.
# Tämä voisi olla esimerkiksi asiakaspalvelussa tikettien luokittelu vakavuusasteikolle
# tai verkkokaupalla käyttäjien arvostelujen luokittelu negatiiviseksi tai positiiviseksi.

# Käytetään tekstin klassifikaatioon Kerasin valmista IMDB elokuva-arvostelu datasettiä.
# Datasetissä arvostelut ovat listoja numeroista, joita vastaavat sanat löytää datasetin sanaindeksistä.
# Sanaindeksi on muodustettu niin, että mitä useammin sana esiintyy, sitä pienempi sen indeksi on.
# Sanaindeksissä indeksit 0-4 on varattu muita merkkejä varten.

# Yleensä oikean maailman tapauksissa näin hyvin tekstiä ei ole esikäsitelty,
# joten muutetaan alla kaikki arvostelut ensin tekstiksi.

# Indeksit tekstiksi: https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset/44891281
import tensorflow as tf
# Kuinka monta sanaa sanakirjaan tallennetaan
num_words = 20000
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.imdb.load_data(seed=1, num_words=num_words)
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
print(train_reviews[1])

'''
runcell(4, 'R:/Koodaus/repos/jamk/data_analytics/syvaoppiminen/koodit/osa3/nlp.py')
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17465344/17464789 [==============================] - 13s 1us/step
<__array_function__ internals>:5: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
P:\Programs\Anaconda3\envs\syvaoppiminen\lib\site-packages\tensorflow\python\keras\datasets\imdb.py:159: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
   8192/1641221 [..............................] - ETA: 0sP:\Programs\Anaconda3\envs\syvaoppiminen\lib\site-packages\tensorflow\python\keras\datasets\imdb.py:160: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
  x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
1646592/1641221 [==============================] - 1s 1us/step

<START> after several extremely well ratings to the point of superb i was extremely pleased with the film
 the film was dark moving the anger the pain the guilt and a very extremely convincing demon br br i had
 initially expected to see many special effects and like a lover's <UNK> it blew me away with the subtlety
 and the <UNK> of it brian i am again blown away with your artistry with the telling of the story and your
 care of the special effects you will go a long way my friend i will definitely be the president of your fan
 club br br eric <UNK> the best actor award was the number one choice you made jr lopez look like a child compared
 to <UNK> br br overall the acting story line the high quality filming and awesome effects it was fantastic i just
 wish it were longer i am looking forward to the <UNK> with extremely high expectations
'''
# Näemme koulutusdatan toisesta arvostelusta,
# että lauseen alku on määritetty 'START' tokenilla ja harvoin esiintyvät sanat,
# jotka eivät siis olleet 20 000 useimpien esiintyvien sanojen joukossa, on korvattu 'UNK' tokenilla.

#%%
# Nyt kun teksti on kuvitteellisesti enemmän sellaisessa muodossa, minkälaisena sen löytäisi arkielämässä, voidaan lähteä luomaan numeerista esitystä käyttäen taas Tokenizer - luokkaa.

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token='<UNK>')
tokenizer.fit_on_texts(train_reviews)

# tekstit sekvenssiksi
train_X = tokenizer.texts_to_sequences(train_reviews)
test_X = tokenizer.texts_to_sequences(test_reviews)

# sekvenssit samankokoiseksi
train_X = tf.keras.preprocessing.sequence.pad_sequences(train_X, padding='post', maxlen=256)
test_X = tf.keras.preprocessing.sequence.pad_sequences(test_X, padding='post', maxlen=256)
print(train_X.shape)
'''
(25000, 256)
'''

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
          epochs=10)

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

# Luodaan testiksi adjektiivejä merkkijonoon.
# Syötetään nämä adjektiivit Embedding - kerrokselle ja otetaan kerroksen ulostulo talteen.
# Näin voidaan visualisoida graafiin se, miten embedding - kerros erottaa adjektiivit toisistaan.

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

#%%

# Tekstin luonti
# Tekstin luonti onnistuu yksinkertaisimmillaan esimerkiksi Markovin ketju - prosessilla.
# Markovin ketjussa tuleva tila riippuu edellisen tilan tiedoista.
# Jos meillä olisi esimerkiksi lauseet:
    # Minä olen Joni Minä olen kurssilla ja Minä tiesin tämän ja lauseet jaettaisiin sanoihin,
# Markovin ketjussa sanan "Minä" jälkeen 66% todennäköisyydellä seuraava sana on "olen"
# ja 33% todennäköisyydellä sana on "tiesin".

# Ladataan datasetti elokuva-arvosteluista GitHub repositoriosta
# https://github.com/nproellochs/SentimentDictionaries,
# poimitaan sieltä kaikki sanat ja yritetään luoda niistä lauseita Markovin ketju - prosessilla.

import pandas as pd
# Poistetaan sanoista tyhjät välilyönnit, pisteet ja pilkut sekä muutetaan kaikki kirjaimet pieniksi
def split_to_words(row):
    row = row.lstrip().rstrip().lower()
    row = row.replace('\n','')
    row = row.replace(',','')
    row = row.replace('.','')
    return row.split(' ')
df = pd.read_csv('https://raw.githubusercontent.com/nproellochs/SentimentDictionaries/master/Dataset_IMDB.csv')
df['Words'] = df['Text'].apply(lambda x: split_to_words(x))
print(df)

'''
runcell(8, 'R:/Koodaus/repos/jamk/data_analytics/syvaoppiminen/koodit/osa3/nlp.py')
         Id  ...                                              Words
0     29420  ...  [in, my, opinion, a, movie, reviewer's, most, ...
1     17219  ...  [starship, troopers, (director:, paul, verhoev...
2     18406  ...  [the, school, of, flesh, (ecole, de, la, chair...
3     18648  ...  [lock, stock, and, two, smoking, barrels, (dir...
4     20021  ...  [run, lola, run, (lola, rennt)(director/writer...
    ...  ...                                                ...
5001   7470  ...  [the, conventional, wisdom, is, that, movie, s...
5002   7853  ...  [nicolas, roeg's, mesmerizing, 1971, film, wal...
5003   8309  ...  [the, movie, air, force, one, should, require,...
5004   8912  ...  ["well, jones, at, least, you, haven't, forgot...
5005   9085  ...  [in, a, time, of, bloated, productions, where,...

[5006 rows x 5 columns]
'''

#%%

import numpy as np
def make_pairs(df):
    for words in df['Words']:
        for i in range(len(words)-1):
            yield (words[i], words[i+1])
# Luodaan generaattorin avulla sanakirja,
# jossa kaikki mahdolliset vaihtoehdot yhdelle lauselle annetaan sanakirjan arvoiksi.
# Aikaisemmasta sanallisesta esimerkistä muodostuisi siis sanakirja: words['Minä'] = ['olen','olen','tiesin']
pairs = make_pairs(df)
words = {}
for word1, word2 in pairs:
    if word1 in words.keys():
        words[word1].append(word2)
    else:
        words[word1] = [word2]
        
# Alustetaan ketju satunnaisella arvostelulla ja poimitaan arvostelusta satunnainen sana
random_row_index = np.random.choice(np.arange(len(df)))
random_row = df['Words'][random_row_index]
random_word_index = np.random.choice(np.arange(len(random_row)))
random_word = random_row[random_word_index]
markov_chain = [random_word]
# Poimitaan sanakirjasta satunnaisesti sana edellisen sanan perusteella, ja lisätään sana lauseeseen.
# Jos sanakirjasta ei löydy sanalle mitään seuraavaa vaihtoehtoa, täytetään ketjua satunnaisella täytesanalla
# Toistetaan tämä, kunnes on luotu lause jossa on 30 sanaa
num_words = 30
for i in range(num_words):
    try:
        markov_chain.append(np.random.choice(words[markov_chain[-1]]))
    except KeyError:
        markov_chain.append(np.random.choice(['the','be','to','of','and','a','in','that','have']))
print(' '.join(markov_chain))

'''
she walks away but might do go north with one facet of her lover is free reviews
 directly by the result is harder edge of his nephew justin) and if the
'''

# Kuten näemme, voimme saada aikaan Markovin ketju - prosessilla melkein ymmärrettäviä lauseita
# tietämällä vain mikä sana seuraa edellistä sanaa.
# Ilmiselvänä ongelmana on, että lauseet eivät seuraa mitään kontekstia eikä arvostelulla ole punaista lankaa.
# Prosessi ei myöskään luo uusia sanoja, vaan käyttää pelkästään datasetissä esiintyviä sanoja.
# Jos Markovin ketjujen kanssa haluaa puuhastella enemmän,
# kannattaa kokeilla markovin ketjujen luomiseen tarkoitettua Python kirjastoa https://github.com/jsvine/markovify.

# Neuroverkoilla voidaan luoda paljon realistisemman näköistä tekstiä.
# Muun muassa OpenAI on julkaissut GPT-2 ja GPT-3 neuroverkkomallit,
# jotka pystyvät luomaan yksittäisestä lauseesta kokonaisen kappaleen verran tekstiä.
# Näiden mallien koulutus sekä tekstin luonti vaativat paljon laskentatehoa.
# Netistä GPT-2 mallin toimintaa pystyy kokeilemaan esimerkiksi Talk To Transformer - sivulta.

#%%

# Lähteet
# Word embedding: https://machinelearningmastery.com/what-are-word-embeddings/
# IMDB ja Embedding kerroksen visualisointi: https://thedatafrog.com/en/word-embedding-sentiment-analysis/
# NLP Kerassissa: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# Markovin ketju: https://towardsdatascience.com/simulating-text-with-markov-chains-in-python-1a27e6d13fc6