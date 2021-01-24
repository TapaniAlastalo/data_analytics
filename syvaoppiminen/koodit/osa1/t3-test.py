import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64') # poistaa TensorFlowin huomautukset datatyypeistä

tensor_rank0 = tf.Variable(1, dtype=tf.int32)
print(tensor_rank0)
tensor_rank1 = tf.Variable([1,2,3], dtype=tf.float32)
print(tensor_rank1)
tensor_rank2 = tf.Variable([[1,2,3],
                           [2,3,4],
                           [3,4,5]], dtype=tf.float64)
print(tensor_rank2)

print(f"Tensorin muoto: {tensor_rank2.shape}")
print(f"Tensorin datatyyppi: {tensor_rank2.dtype}")
print(f"Tensorin arvo: {tensor_rank2.numpy()}")

# esimerkki tensorioperaatio
mat1 = np.array([[1,1],
                 [2,2],
                 [3,3]])
mat2 = np.array([[10,15],
                 [20,25],
                 [30,35]])
add_result = tf.add(mat1,mat2)
print(f"Mat1 ja mat2 matriisien yhteenlaskun tulos: \n{add_result}")

# Matriisin kertolaskussa, jossa matriisien koot ovat (a,b) ja (m,n), b pitää olla yhtä suuri kuin m, jotta kertolaskun voi suorittaa.
# Mat1 ja mat2 ovat muotoa 3 x 2 ja 3 x 2, jolloin b ei ole yhtä suuri kuin m.
# Muotoillaan mat2 uudestaan 2 x 3 matriisiksi, jonka jälkeen kertolasku voidaan suorittaa.

mat2 = tf.reshape(mat2,[mat2.shape[1],mat2.shape[0]])
matmul_result = tf.matmul(mat1,mat2)
print(f"Mat1 ja mat2 matriisien kertolaskun tulos: \n{matmul_result}")


import pandas as pd
# Datasetin kuvauksen voi lukea osoitteesta https://archive.ics.uci.edu/ml/datasets/Iris
# Poimitaan sieltä sarakkeiden nimet Attribute Information otsikon alta
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None,names=['Sepal Length','Sepal Width','Petal Length','Petal Width','Class'])
print(df)
# selvitetään uniikit luokat
df['Class'].unique()
# kategorisoidaan luokat numeerisiksi
df['Class'] = pd.Categorical(df['Class'])
df['Class'] = df['Class'].cat.codes
print(df)


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

# laitetaan data "köntteihin" eli batcheihin. Mitä isompi batch on, sitä enemmän syöttödataa menee kerralla neuroverkon läpi, jolloin neuroverkon koulutus nopeutuu
train_dataset,test_dataset  = train_dataset.batch(1), test_dataset.batch(1)

# keras

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(3) # Kun Denselle ei määritä aktivaatiofunktiota,
                             # sen oletusarvo on 'linear', eli mitään aktivaatiofunktiota
                             # ei suoriteta painoarvojen ja sisääntulon pistetuloon
])

predictions = model(train_X)
print(predictions[1].numpy())

predictions_softmax = tf.nn.softmax(predictions[1])
print(predictions_softmax.numpy())

prediction_argmax = tf.argmax(predictions,axis=1)
print(prediction_argmax[1])

print("Ennuste:")
print(train_y[1])

# luodaan keras kutsun sisääntulokerroksen, piilotettu kerroksen ja ulostulokerroksen aktivaatiofunktiot.
#Luodaan neuroverkko uudestaan niin, että sen sisällä on myös ulostulokerros, jossa on softmax - aktivaatiofunktio. Tällöin emme tarvitse aikaisempia toimenpiteitä luokan poimimiseen ennusteesta, vaan neuroverkko antaa sen suoraan. Ulostulokerrokseen tarvitaan niin monta neuronia, kuin on ennustettavia luokkia, eli tässä tapauksessa kolme.
#Jos ennustettavia luokkia olisi vain kaksi (luokka x ja luokka y), käyttäisimme vain yhtä neuronia ja sigmoid - aktivaatiofunktiota. Tällöin ulostulon arvo lähellä nollaa ennustaisi luokkaa x ja arvo lähellä ykköstä ennustaisi luokkaa y
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dense(8,activation='relu'), # lisätään toiseen piiloitettuun kerrokseen relu - aktivaatiofunktio.
    tf.keras.layers.Dense(3,activation='softmax') # ulostulokerros, jossa softmax - aktivaatiofunktio
])

# määritellään optimisaattori, virhefuntio ja koulutuksenaikainen seuranta
model.compile(optimizer='adam', # Adam optimisaattori, joka alustetaan oletusparametreillä
              loss='sparse_categorical_crossentropy', # Virhefunktio
              metrics=['accuracy'] # Seurataan koulutuksen aikana, kuinka neuroverkon tarkkuus muuttuu
             )

# sovitetaan opetusdata
model.fit(train_dataset, 
          epochs=20)
# Jos ei käytä TensorFlowin Dataset - luokkaa:
# train_X = np.array([1,2,3])
# train_y = np.array([4,5,6])
# model.fit(train_X,
#          train_y,
#          epochs=10)

# testataan mallia
results = model.evaluate(test_dataset)
# Jos ei käytä TensorFlowin Dataset - luokkaa:
# test_X = np.array([1,2,3])
# test_y = np.array([4,5,6])
# results = model.evaluate(test_X,
#                          test_y)
print(results)

print(f"Neuroverkon tarkkuus testiarvoihin verrattuna: {results[1]*100:.0f}%")




