#import numpy as np
#import pandas as pd
#import sklearn
#import tensorflow as tf

#print(tf.__version__) # Pitäisi tulostaa tensorflow 2.0 tai uudempi versio

# only when using gpu version
#tf.test.is_gpu_available(cuda_only=True)

#%%

# Aktivaatiofunktiot: https://medium.com/ai%C2%B3-theory-practice-business/a-beginners-guide-to-numpy-with-sigmoid-relu-and-softmax-activation-functions-25b840a9a272
# Neuroverkon luonti NumPyllä: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# Neuroverkon rakenteet kuvitettu käyttäen työkalua: http://alexlenail.me/NN-SVG/index.html
# Backpropagation laskukaavat: https://developers-dot-devsite-v2-prod.appspot.com/machine-learning/crash-course/backprop-scroll ja https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python ja https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ sekä https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf#48c4
# Suosittelen katsomaan alla olevan Youtube soittolistan, jossa on hyvin visualisoitu matematiikka neuroverkkojen taustalla: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

#%%

# Tensorflow 2.0
# Tensorflow on Googlen kehittämä neuroverkkojen koulutukseen tarkoitettu ohjelmistokehys.
# Tensorflowista on julkaistu äskettäin 2.0 versio, joka esitetään tässä notebookissa.
# Muita neuroverkko-ohjelmistokehyksiä ovat Facebookin PyTorch tai Apachen MXNet.
# Tensorflowissa opitut perusteet neuroverkkojen koulutuksesta on helppo siirtää esimerkiksi PyTorchiin,
# jos ohjelmistokehystä haluaa vaihtaa.

# Tensori
# Tensori on moniulotteinen taulukko.
# Tensorflowissa tensorit toimivat samalla tavalla kuin NumPyn taulukot, eli tensoreilla on datatyyppi ja muoto.
# Tensoreissa on se hyöty, että toisin kuin NumPy taulukkoja,
# ne voidaan viedä näytönohjaimen muistiin laskettavaksi, jolloin saadaan paljon enemmän laskentatehoa käyttöön.

# Tuodaan tarvittavat kirjastot
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64') # poistaa TensorFlowin huomautukset datatyypeistä

#%%
# Tensoreita voi luoda esim. Variable - luokan avulla.
# Normaalisti tensorin arvo on muokkaamaton, paitsi nimensä mukaisesti Variable luokalla tensorin luodessa.
# Luokan kutsun sisään määritellään tensorin arvo ja datatyyppi.
# Käytetään Variable - luokkaa luomaan yllä luetellut tensorit:

tensor_rank0 = tf.Variable(1, dtype=tf.int32)
print(tensor_rank0)
tensor_rank1 = tf.Variable([1,2,3], dtype=tf.float32)
print(tensor_rank1)
tensor_rank2 = tf.Variable([[1,2,3],
                           [2,3,4],
                           [3,4,5]], dtype=tf.float64)
print(tensor_rank2)
'''
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=1>
<tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>
<tf.Variable 'Variable:0' shape=(3, 3) dtype=float64, numpy=
array([[1., 2., 3.],
       [2., 3., 4.],
       [3., 4., 5.]])>
'''

#%%
# Tensorista voi attribuuttien avulla tulostaa tietoja,
# kuten tensorin muoto, datatyyppi ja sisältö.
# Näihin pääsee käsiksi vastaavilla attribuuteilla:

print(f"Tensorin muoto: {tensor_rank2.shape}")
print(f"Tensorin datatyyppi: {tensor_rank2.dtype}")
print(f"Tensorin arvo: {tensor_rank2.numpy()}")

'''
Tensorin muoto: (3, 3)
Tensorin datatyyppi: <dtype: 'float64'>
Tensorin arvo: [[1. 2. 3.]
 [2. 3. 4.]
 [3. 4. 5.]]
'''

#%%
# Esimerkki tensorioperaatioista, kuten matriisiyhteenlaskusta ja matriisikertolaskusta:

mat1 = np.array([[1,1],
                 [2,2],
                 [3,3]])
mat2 = np.array([[10,15],
                 [20,25],
                 [30,35]])
add_result = tf.add(mat1,mat2)
print(f"Mat1 ja mat2 matriisien yhteenlaskun tulos: \n{add_result}")

# Matriisin kertolaskussa, jossa matriisien koot ovat (a,b) ja (m,n),
# b pitää olla yhtä suuri kuin m, jotta kertolaskun voi suorittaa.
# Mat1 ja mat2 ovat muotoa 3 x 2 ja 3 x 2, jolloin b ei ole yhtä suuri kuin m.
# Muotoillaan mat2 uudestaan 2 x 3 matriisiksi, jonka jälkeen kertolasku voidaan suorittaa.

mat2 = tf.reshape(mat2,[mat2.shape[1],mat2.shape[0]])
matmul_result = tf.matmul(mat1,mat2)
print(f"Mat1 ja mat2 matriisien kertolaskun tulos: \n{matmul_result}")

'''
Mat1 ja mat2 matriisien yhteenlaskun tulos: 
[[11 16]
 [22 27]
 [33 38]]
Mat1 ja mat2 matriisien kertolaskun tulos: 
[[ 35  45  55]
 [ 70  90 110]
 [105 135 165]]
'''

#%%
# Ladataan tällä kertaa oikean maailman datasetti, jonka avulla voimme kouluttaa neuroverkon.
# Yksi suosituimmista dataseteistä on kukkadatasetti,
# jossa on annettu Iris setosa, Iris versicolor ja Iris virginica kukista terälehtien ja verholehden pituudet.
# Tavoitteena on luoda neuroverkkomalli, joka terälehden ja versolehden pituuksien perusteella ennustaa,
# mikä kukka on kyseessä.

# Käytetään datasetin lukemiseen aikaisemmilta kursseilta tuttua Pandas - kirjastoa,
# jonka avulla voi näppärästi lukea CSV tiedostoja suoraan linkeistä.

import pandas as pd
# Datasetin kuvauksen voi lukea osoitteesta https://archive.ics.uci.edu/ml/datasets/Iris
# Poimitaan sieltä sarakkeiden nimet Attribute Information otsikon alta
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None,names=['Sepal Length','Sepal Width','Petal Length','Petal Width','Class'])
print(df)

'''
runcell(6, 'R:/Koodaus/repos/jamk/data_analytics/syvaoppiminen/koodit/osa1/t0.py')
     Sepal Length  Sepal Width  Petal Length  Petal Width           Class
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
3             4.6          3.1           1.5          0.2     Iris-setosa
4             5.0          3.6           1.4          0.2     Iris-setosa
..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica

[150 rows x 5 columns]
'''

#%%
# Katsotaan vielä Pandasilla, mitä eri kukkia datasetissä on, ottamalla "Class" - sarakkeesta uniikit arvot.

print(df['Class'].unique())
'''
['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
'''

#%%
df['Class'] = pd.Categorical(df['Class'])
df['Class'] = df['Class'].cat.codes
print(df)
'''
runcell(8, 'R:/Koodaus/repos/jamk/data_analytics/syvaoppiminen/koodit/osa1/t0.py')
     Sepal Length  Sepal Width  Petal Length  Petal Width  Class
0             5.1          3.5           1.4          0.2      0
1             4.9          3.0           1.4          0.2      0
2             4.7          3.2           1.3          0.2      0
3             4.6          3.1           1.5          0.2      0
4             5.0          3.6           1.4          0.2      0
..            ...          ...           ...          ...    ...
145           6.7          3.0           5.2          2.3      2
146           6.3          2.5           5.0          1.9      2
147           6.5          3.0           5.2          2.0      2
148           6.2          3.4           5.4          2.3      2
149           5.9          3.0           5.1          1.8      2

[150 rows x 5 columns]
'''

#%%

# Nyt kukka Iris-setosa on 0, Iris-versicolor 1 ja Iris-virginica 2.

# Luodaan tensorflow Dataset luokka, joka on Tensorflowin luoma data pipeline,
# joka yksinkertaistaa ja nopeuttaa datan lukemista.

# Data on järjestyksessä, sekoitetaan, jotta saadaan koulutus- ja testidataan eri classeja
df = df.sample(frac=1)

# Jaetaan datasetti ominaisuuksiin ja ennustettavaan muuttujaan (features and labels)
# Tässä tapauksessa pituudet ovat ominaisuuksia ja kukan numeerinen luokka on ennustettava muuttuja
# Muutetaan .values attribuutilla DataFramet helposti NumPy taulukoiksi
train_X, test_X = df.drop('Class',axis=1)[0:130].values, df.drop('Class',axis=1)[130:].values
train_y, test_y = df['Class'][0:130].values, df['Class'][130:].values

# Muutetaan from_tensor_slices - funktiolla  NumPy - taulukot TensorFlowin Dataset - luokkaan
train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_X,test_y))

# laitetaan data "köntteihin" eli batcheihin. Mitä isompi batch on,
# sitä enemmän syöttödataa menee kerralla neuroverkon läpi, jolloin neuroverkon koulutus nopeutuu
train_dataset,test_dataset  = train_dataset.batch(1), test_dataset.batch(1)

#%%
# KERAS

# Keras on korkeamman tason rajapinta, jonka avulla yksinkertaistetaan neuroverkon luontia ja koulutusta TensorFlowissa.

# Kerassin Sequential luokalla voi luoda neuroverkon määrittelemällä sille listassa neuroverkon kerrokset,
# joita haluaa käyttää. Kerassissa kerros tarkoittaa joukkoa neuroneita,
# joilla kaikilla on sama aktivaatiofunktio, mikä on kerroksen luonnissa määritelty.
# Listan ensimmäinen elementti on neuroverkon ensimmäinen kerros,
# ja listan viimeinen elementti neuroverkon ulostulokerros.

# Yleisin kerros, mitä Kerassissa käytetään, on ns. Dense - kerros.
# Kerroksessa jokaisen edellisen kerroksen neuroni yhdistyy kaikkiin seuraavan kerroksen neuroneihin,
# aivan kuten aikaisemmissa esimerkeissä.
# Densen parametreinä määritellään neuroneiden määrä sekä neuroneiden aktivaatiofunktio.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(3) # Kun Denselle ei määritä aktivaatiofunktiota,
                             # sen oletusarvo on 'linear', eli mitään aktivaatiofunktiota
                             # ei suoriteta painoarvojen ja sisääntulon pistetuloon
])

#%%
# Vaikka neuroverkkoa ei ole vielä koulutettu, voimme luoda mallilla ennusteen antamalla sille dataa parametrinä.

predictions = model(train_X)
print(predictions[1].numpy())
'''
[-1.6767733  -0.12134068  0.80389226]
'''

#%%
# Ennusteena saadaan jokaiselle luokalle ns. logit arvot. Softmax - aktivaatiofunktion avulla logit arvot voidaan skaalata välille 0-1.

predictions_softmax = tf.nn.softmax(predictions[1])
print(predictions_softmax.numpy())
'''
[0.05654078 0.26784133 0.6756179 ]
'''

#%%
# Tällöin se luokka, jonka arvo on lähinnä ykköstä, on neuroverkon ennustus syöttödatan luokalle.
# Voimme käyttää vielä TensorFlowin argmax funktiota,
# joka palauttaa meille suurimman arvon saaneen luokan ennustuksista.
prediction_argmax = tf.argmax(predictions,axis=1)
print(prediction_argmax[1])
'''
tf.Tensor(2, shape=(), dtype=int64)
'''

#%%
# Neuroverkko ennusti luokaksi siis 1, eli 'Iris-versicolor'. Voimme katsoa train_y muuttujasta, mikä kukan luokka oikeasti oli:

train_y[1]
# 2

#%%
# Luokka oli oikeasti 2 eli 'Iris-virginica'. On ymmärrettävää, että ennustus ei mennyt kohdilleen, koska neuroverkkoa ei koulutettu ollenkaan vaan ennustus tapahtuu nyt täysin satunnaisesti määrätyillä painoarvoilla.

# Luodaan neuroverkko uudestaan niin, että sen sisällä on myös ulostulokerros,
# jossa on softmax - aktivaatiofunktio.
# Tällöin emme tarvitse aikaisempia toimenpiteitä luokan poimimiseen ennusteesta,
# vaan neuroverkko antaa sen suoraan. Ulostulokerrokseen tarvitaan niin monta neuronia,
# kuin on ennustettavia luokkia, eli tässä tapauksessa kolme.

# Jos ennustettavia luokkia olisi vain kaksi (luokka x ja luokka y),
# käyttäisimme vain yhtä neuronia ja sigmoid - aktivaatiofunktiota.
# Tällöin ulostulon arvo lähellä nollaa ennustaisi luokkaa x ja arvo lähellä ykköstä ennustaisi luokkaa y.

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(8,activation='relu'), # lisätään toiseen piiloitettuun kerrokseen relu - aktivaatiofunktio.
    tf.keras.layers.Dense(3,activation='softmax') # ulostulokerros, jossa softmax - aktivaatiofunktio
])

#%%
# Kerassin Sequential - luokka periytyy Model - luokasta, josta se perii mm. metodit compile ja fit.
# Model - luokan dokumentaatio: https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=stable

# Ennen kuin neuroverkko voidaan kouluttaa, tulee compile - funktion parametreillä määrittää koulutuksessa
# käytettävä optimisaattori ja virhefunktio. Metrics - parametriin voi luoda listan arvoista,
# joiden kehitystä haluaa seurata koulutuksen aikana.
# Listassa voisi olla vaikka tarkkuus eli accuracy, joka kertoo,
# kuinka suuri osa neuroverkon ennustuksista meni oikein.

model.compile(optimizer='adam', # Adam optimisaattori, joka alustetaan oletusparametreillä
              loss='sparse_categorical_crossentropy', # Virhefunktio
              metrics=['accuracy'] # Seurataan koulutuksen aikana, kuinka neuroverkon tarkkuus muuttuu
             )

#%%
# Koulutus tapahtuu kutsumulla fit - funktiota. Parametreinä tulee määrittää datasetti,
# jonka avulla neuroverkko koulutetaan.
# Vastaavasti jos datasettiä ei ole luotu Tensorflowin Dataset - luokan avulla,
# funktiolle tulee ensin syöttää ominaisuudet (train_X)
# ja tämän jälkeen ennustettavat arvot (train_y) omissa muuttujissaan.
# Muuttujat tulee syöttää myös NumPy - taulukkoina

# Epoch - parametri kertoo sen, kuinka monta kertaa datasetti käydään läpi.
# Jos datasetti on iso, niin parametrin arvo kannattaa asettaa välille 5-20.
# Pienemmissä dataseteissä parametrin arvo voi olla väliltä 50-100.

model.fit(train_dataset, 
          epochs=20)
# Jos ei käytä TensorFlowin Dataset - luokkaa:
# train_X = np.array([1,2,3])
# train_y = np.array([4,5,6])
# model.fit(train_X,
#          train_y,
#          epochs=10)

#%%

# Evaluate - funktion avulla voi verrata neuroverkon ennustuksia oikeisiin arvoihin.
# Antamalla tälle funktiolle parametrinä testidatasetin,
# funktio automaattisesti syöttää neuroverkolle test_X eli testiominaisuudet
# ja vertailee neuroverkon ennustuksia oikeisiin arvoihin eli test_y arvoihin.

results = model.evaluate(test_dataset)
# Jos ei käytä TensorFlowin Dataset - luokkaa:
# test_X = np.array([1,2,3])
# test_y = np.array([4,5,6])
# results = model.evaluate(test_X,
#                          test_y)
print(results)
'''
20/20 [==============================] - 0s 500us/step - loss: 0.2039 - accuracy: 0.9500
[0.2038700915290974, 0.95]
'''

#%%
# Funktio palauttaa listan, jossa ensimmäinen arvo kertoo virhefunktion arvon testidatassa.
# Listan toinen elementti kertoo tarkkuuden, eli kuinka monta ennustusta neuroverkolla meni
# oikein jaettuna kaikilla testiarvoilla. Tämän arvon voi kertoa sadalla,
# jolloin saadaan neuroverkon tarkkuus prosentteina.

print(f"Neuroverkon tarkkuus testiarvoihin verrattuna: {results[1]*100:.0f}%")
# Neuroverkon tarkkuus testiarvoihin verrattuna: 95%

#%%
'''
Yhteenveto
Ladattiin Pandasin avulla CSV-tiedostosta kukkien mittoja DataFrameen
Tehtiin koulutus- ja testijako
Käytettiin Tensorflowin Dataset luokkaa
Luotiin neuroverkko, jossa on yhteensä neljä kerrosta
Sisääntulokerros, jota harvemmin neuroverkon luonnissa tarvitsee määritellä
Ensimmäinen piiloitettu kerros, jossa on 4 neuronia ja aktivaatiofunktiona Rectified Linear Unit eli ReLU
Toinen piiloitettu kerros, jossa on 3 neuronia ja aktivaatiofunktiona myös ReLU
Ulostulokerros, jossa on aktivaatiofunktiona softmax
Määritettiin optimisaatiomenetelmäksi Adam ja häviöfunktioksi _sparse_categoricalcrossentropy
Koulutettiin neuroverkko koulutusdatasetillä, joka käytiin läpi yhteensä kaksikymmentä kertaa
Määritettiin neuroverkon tarkkuus testidatasetin avulla.
'''