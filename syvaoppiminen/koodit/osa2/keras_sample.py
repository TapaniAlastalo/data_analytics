# ladataan tensorin valmista regressiodataa
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
boston = tf.keras.datasets.boston_housing

# jaetaan datasetti
(train_features, train_labels), (test_features, test_labels) = boston.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))

# sekoitetaan datasetti ja jaetaan 8 rivin batcheihin
train_dataset, test_dataset = train_dataset.shuffle(10).batch(8), test_dataset.shuffle(10).batch(8)

#%%
# sisääntulokerroksen määritys
# tarkastellaan missä muodossa data syötetään neuroverkkoon
print(train_features.shape)
# (404, 13) eli saapuu 404 riviä ja 13 saraketta
# Tälä mallilla määritetään sarakkeiden määrä shape=(13)
input_layer = tf.keras.Input(shape=(13))
print(input_layer.shape)

#%%
# piilotetut kerrokset
# syntaksi: uusi_kerros = uusi_kerros_luokka(parametrit)(edellinen_kerros)
dense_1 = tf.keras.layers.Dense(10, activation='relu')(input_layer)
# määritellään toinen piilotettu kerros
dense_2 = tf.keras.layers.Dense(20, activation='relu')(dense_1)
# määritellään ulostulokerros. ennustetaan vain yhtä arvoa, joten määritetään vain yksi neuroni
output_layer = tf.keras.layers.Dense(1, activation='linear')(dense_2)

# yhdistetään kerrokset neuroverkoksi Keras Model luokalla. 
# Määrittelyyn riittää sisääntulo ja ulostulokerrokset, sillä jokainen kerros sisältää edeltävän kerroksen
model_functional = tf.keras.Model(inputs=input_layer,
                                  outputs=output_layer)
# tulostetaan neuroverkon rakenne
model_functional.summary()

#%%
# varmistetaan mallin toiminta kouluttamalla se
model_functional.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # 'adam',
                         loss='mean_squared_error', # Käytetään jäännösneliösummaa. Regressio ongelmissa tarkkuutta (accuracy) ei voida mitata. 
                         metrics=['mean_squared_error'])
# varsinainen opetus (kerrat)
model_functional.fit(train_dataset,
                     epochs=10)

#%%
# neuroverkkojen visualisointi 
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

tf.keras.utils.plot_model(model_functional,dpi=70,show_shapes=True)

#%%
# neuroverkkomallien kutsuminen kerroksittain
nn_x_input = tf.keras.Input(shape=(13))
nn_x_dense1 = tf.keras.layers.Dense(10,activation='relu')(nn_x_input)
nn_x_dense2 = tf.keras.layers.Dense(20,activation='relu')(nn_x_dense1)
nn_x_output = tf.keras.layers.Dense(10,activation='linear')(nn_x_dense2)

nn_x = tf.keras.Model(inputs=nn_x_input, 
                      outputs=nn_x_output)

# Luodaan neuroverkko Y
nn_y_input = tf.keras.Input(shape=(10)) # sisääntulon muoto yhtä suuri kuin neuronien määrä X:n ulostulossa
nn_y_dense1 = tf.keras.layers.Dense(10,activation='relu')(nn_y_input)
nn_y_dense2 = tf.keras.layers.Dense(20,activation='relu')(nn_y_dense1)
nn_y_output = tf.keras.layers.Dense(1,activation='linear')(nn_y_dense2)

nn_y = tf.keras.Model(inputs=nn_y_input, 
                      outputs=nn_y_output)

# Yhdistetään kaksi neuroverkkoa, jossa neuroverkko Y ottaa neuroverkon X ulostuloarvon sisääntuloarvona
model_input = tf.keras.Input(shape=(13)) # Luodaan sisääntulo koko neuroverkolle
nn_x_out = nn_x(model_input) # Määritetään koko neuroverkon sisääntulo menemään neuroverkko X sisääntuloon.
nn_y_out = nn_y(nn_x_out) # Määritetään neuroverkko X:n ulostulo menemään neuroverkko Y:n sisääntuloksi

combined_nn = tf.keras.Model(inputs=model_input, 
                             outputs=nn_y_out)
combined_nn.summary()

#%%
# Yhdistetyn mallin summary - funktiosta nähdään, että mallin sisällä on alamallit model_1 ja model_2. Antamalla model_to_dot funktiolle parametri 'expand_nested = True' voidaan tulostaa model_1 ja model_2 mallien rakenteet.
tf.keras.utils.plot_model(combined_nn,dpi=70,show_shapes=True,expand_nested=True) # Näytetään kuva suoraan notebookissa


#%%
# Kokoelma (Ensemble) - mallin luominen
# Esim. koneoppimisen puolella satunnaismetsä - menetelmässä luodaan ennustuksia laskemalla yksittäisten päätöspuiden päätösten keskiarvo. Samaa ns. kokoelma - menetelmää voitaisiin käyttää myös neuroverkkojen kanssa, eli luodaan yhden ongelman ratkaisuun monta neuroverkkoa ja lasketaan näiden ennustuksien keskiarvo, jolloin saadaan koko kokoelman ennustus.
# Tavoitteena on luoda eri neuroverkkorakenteita, jotka pyrkivät havainnoimaan datasetistä eri ominaisuuksia. Jotta kokoelma - malli toimisi, niin jokaisen neuroverkon ulostuloarvot on oltava samassa skaalassa ja muodossa, muuten keskiarvon laskeminen niistä ei onnistu.

# Ensimmäisen mallin määritys
model1_input = tf.keras.Input(shape=(13)) 
model1_dense = tf.keras.layers.Dense(10,activation='relu')(model1_input)
model1_output = tf.keras.layers.Dense(1,activation='linear')(model1_dense)
model1 = tf.keras.Model(inputs=model1_input,
                        outputs=model1_output)
# Toisen mallin määritys
model2_input = tf.keras.Input(shape=(13)) 
model2_dense1 = tf.keras.layers.Dense(20,activation='relu')(model2_input)
model2_dense2 = tf.keras.layers.Dense(10,activation='relu')(model2_dense1)
model2_output = tf.keras.layers.Dense(1,activation='linear')(model2_dense2)
model2 = tf.keras.Model(inputs=model2_input,
                        outputs=model2_output)
# Kolmannen mallin määritys
model3_input = tf.keras.Input(shape=(13))
model3_output = tf.keras.layers.Dense(1,activation='linear')(model3_input)
model3 = tf.keras.Model(inputs=model3_input,
                        outputs=model3_output)
# Kootaan kaikki yhteen kokoelma - malliin
ensemble_input = tf.keras.Input(shape=(13)) # koko kokoelma - mallille tuleva sisääntulokerros
out1 = model1(ensemble_input)
out2 = model2(ensemble_input)
out3 = model3(ensemble_input)
ensemble_output = tf.keras.layers.average([out1,out2,out3])
ensemble_model = tf.keras.Model(inputs=ensemble_input,
                                outputs=ensemble_output)
# Piirretään kuva kokoelmasta
tf.keras.utils.plot_model(ensemble_model,dpi=70,show_shapes=True,expand_nested=True)

# Näemme kuvasta, että sisääntulo syötetään jokaiselle neuroverkolle, josta neuroverkot laskevat jokainen oman ennusteensa. Nämä ennusteet kootaan Average - kerroksessa yhdeksi ennusteeksi.


#%%
# Epälineaaristen neuroverkkojen luominen

# Jaetaan koulutusdatan ominaisuudet ensin kahteen osaan. Tässä on käytössä notebookin alussa ladattu Bostonin asuntojen hintadatasetti.
train_features_input1, train_features_input2 = train_features[:,:2], train_features[:,2]
# Tulostetaan ominaisuuksien muodot, jotta tiedetään, mitkä muodot määritetään neuroverkon sisääntuloihin.
print(train_features_input1.shape)
print(train_features_input2.shape)

# Huomataan, että input1:ssä on 404 riviä ja kaksi saraketta ja input2:ssa 404 riviä ja yksi sarake.
# Luodaan input1 ja input2 sisääntulokerrokset tämän tiedon avulla:
input1 = tf.keras.Input(shape=(2,))
input2 =tf.keras.Input(shape=(1,))
dense1 = tf.keras.layers.Dense(2,activation='relu')(input1) # Luodaan kuvassa olevat "Dense1_1" ja "Dense1_2" neuronit.

# dense1 kerroksen ulostulo ja input2 sisääntulo tulee yhdistää, ennen kuin sen voi syöttää eteenpäin Dense2_1, Dense2_2 sekä Dense2_3 neuroneille.
# Käytetään tätä varten concatenate - kerrosta.
concat = tf.keras.layers.concatenate([dense1,input2]) # Määritellään yhdistettävät sisääntulot listan avulla.
dense2 = tf.keras.layers.Dense(3,activation='relu')(concat)
output_layer = tf.keras.layers.Dense(1,activation='sigmoid')(dense2)

model_complex = tf.keras.Model(inputs=[input1,input2], # Määritellään useampi sisääntulo neuroverkolle käyttäen listaa.
                               outputs=output_layer)
tf.keras.utils.plot_model(model_complex,dpi=70,show_shapes=True)
# Alussa kuvitettu neuroverkon rakenne ja model_to_dot funktion luoma kaava eroavat toisistaan sen verran,
# että lopputuloksessa on yksi nyt ylimääräinen "Concatenate" kerros, jossa yhdistämme Dense kerroksen ja toisen sisääntulokerroksen syötteet yhdeksi, jotta voimme syöttää sen eteenpäin seuraavalle Dense kerrokselle. 


#%%
# Neuroverkkoluokka
class CustomModel(tf.keras.Model):
    def __init__(self):
        # Super - funktiolla peritään Kerassin Model luokasta tarpeelliset funktiot (kuten compile, fit)
        super(CustomModel, self).__init__()
        # luodaan neuroverkon kerrokset
        self.d1 = tf.keras.layers.Dense(10, activation='relu')
        self.d2 = tf.keras.layers.Dense(20, activation='relu')
        self.d3 = tf.keras.layers.Dense(1, activation='linear')
    # Määritellään, missä järjestyksessä neuroverkon kerrokset käydään läpi call - funktiolla
    def call(self, x):
        x = self.d1(x) # ensimmäinen piiloitettu kerros
        x = self.d2(x) # toinen piiloitettu kerros
        out = self.d3(x) # viimeinen ulostulokerros
        return out # palauta ulostulokerroksen arvo
    
model_class = CustomModel()
model_class.build(input_shape=(None,13)) # Tarvitaan build - funktio, johon määritellään sisääntulon muoto, jotta voidaan tulostaa summary - funktiolla rakenne.
model_class.summary()


#%% Koulutettujen mallien tallennus ja lataaminen tiedostosta
# tallennus
model_functional.save('model_functional.h5')
# lataaminen
model_functional_from_file = tf.keras.models.load_model('model_functional.h5')
# tulosta yhteenveto
model_functional_from_file.summary()


#%%

# Vaihtehtoisia tapajo kouluttaa neuroverkkoja
# Yksi vaihtoehto olisi käyttää TensorFlowin tf.GradientTape API:a.
# GradientTape API automatisoi differentiaalilaskut ja mahdollistaa gradienttien arvojen hakemisen.
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None,names=['Sepal Length','Sepal Width','Petal Length','Petal Width','Class'])
df['Class'] = pd.Categorical(df['Class'])
df['Class'] = df['Class'].cat.codes
train_X, test_X = df.drop('Class',axis=1)[0:130].values, df.drop('Class',axis=1)[130:].values
train_y, test_y = df['Class'][0:130].values, df['Class'][130:].values
train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_X,test_y))
train_dataset,test_dataset  = train_dataset.shuffle(10000).batch(1), test_dataset.shuffle(10000).batch(1)

# luodaan pieni neuroverkko Sequential luokasta
model = tf.keras.Sequential([
    tf.keras.Input(shape=(train_X.shape[1])),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])
model.summary()
#%%
# Määritellään kouluttamista varten virhe- ja gradienttifunktiot.
# Mitä isompi virhefunktion arvo on, sitä huonompi neuroverkko on.
# Tavoitteena on pienentää virhettä, ja näin parantaa neuroverkon ennustuksien tarkkuutta.
# Virhefunktiona voidaan käyttää TensorFlowin luokkaa SparseCategoricalCrossentropy

# Virhefunktion luominen ja virheen laskeminen ennusteesta:
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def calculate_loss(model, features,labels):
    predictions = model(features)
    loss = loss_function(y_true = labels,
                         y_pred = predictions)
    return loss

# Luodaan funktio gradientin laskua varten:
def calculate_gradients(model,features,label):
    with tf.GradientTape() as tape:
        loss = calculate_loss(model,features,label)
    return loss, tape.gradient(loss,model.trainable_variables)

# Nyt kun meillä on virhe- ja gradienttifunktiot tehty, tarvitsemme optimisaattorifunktion.
# Tämän avulla lasketaan virhe ja gradienttien perusteella,
# mihinkä suuntaan ja kuinka paljon painoarvoja tulisi muokata (negatiiviseen tai positiiviseen suuntaan).
# Painoarvoja päivitetään oppimisnopeuden (learning rate) mukaan.
# Optimisaattori voidaan luoda esim. TensorFlowin Adam luokasta.

optimizer = tf.keras.optimizers.Adam(lr=0.001) # määritellään oppimisnopeus lr - parametrissä optimisaattorin luonnin yhteydessä.
# Laske virhe ja gradientti syötteelle
loss, grads = calculate_gradients(model,train_X,train_y)
# Esimerkki virheestä, kun neuroverkkoa ei ole koulutettu vielä yhtään
print(f"Step: {optimizer.iterations.numpy()} Loss: {loss.numpy()}")
# Kuinka virhe pienenee, kun yksi könttä on viety neuroverkon läpi ja painoarvot päivitetty
optimizer.apply_gradients(zip(grads, model.trainable_variables))
print(f"Step: {optimizer.iterations.numpy()} Loss: {calculate_loss(model, train_X, train_y).numpy()}")

# Step: 0 Loss: 1.102077841758728
# Step: 1 Loss: 1.099433422088623

#%%
# Voimme näiden työkalujen avulla nyt kouluttaa neuroverkon.
# Kun neuroverkolle syötetään koko datasetti kerran läpi, kutsutaan sitä yhdeksi epookiksi (epoch).
# Yleensä on syytä käydä koko datasetti useamman kerran läpi,
# jotta neuroverkko ehtii päivittämään painoarvot ja löytämään virhefunktion minimin.
# On syytä luoda siis koulutuskierrokset, joita käydään läpi useampi kerta.
losses = []
accuracies = []
epochs = 10
# Käydään koulutusdata 10 kertaa läpi
for epoch in range(1,epochs+1):
    epoch_loss = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # käy yksi epoch läpi
    for feature, label in train_dataset:
        loss, gradients = calculate_gradients(model,feature,label)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss(loss)
        prediction = model(feature)
        epoch_accuracy(label,prediction)
    losses.append(epoch_loss.result())
    accuracies.append(epoch_accuracy.result())
    print(f"Epoch {epoch:01d} Loss {epoch_loss.result():.2f} Accuracy {epoch_accuracy.result()*100:.2f}%")
    
# Epoch 10 Loss 0.64 Accuracy 96.15%

#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Luodaan kaksi alikuvaajaa
fig, axes = plt.subplots(2,sharex=True,figsize=(14,10))
# Määritetään ensimmäiseen alikuvaajaan virheen kehitys
axes[0].set_ylabel("Loss")
axes[0].plot(losses)
# Määritetään toiseen alikuvaajaan tarkkuuden kehitys
axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Epochs")
axes[1].plot(accuracies)
plt.show()

