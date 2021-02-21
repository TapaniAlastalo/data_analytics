from IPython.display import Image
import os
os.environ['PATH'] = os.environ['PATH']+';'+ os.environ['CONDA_PREFIX'] + r"\pkgs\graphviz-2.38-hfd603c8_2\Library\bin\graphviz"

# Takaisinkytkeytyvä neuroverkko eli Recurrent neural network (RNN)
# RNN mallien avulla voidaan hyödyntää tietoa menneisyydestä.
# RNN - rakenteita kannattaakin käyttää sellaisten ongelmien ratkaisuun,
# kun menneillä arvoilla on vaikutusta tuleviin arvoihin, kuten aikasarjaennustamisessa tai tekstin käsittelyssä.
# Tämä onnistuu syöttämällä neuroverkon seuraavalle askeleelle tietoa aiemmista askeleista.

# Lähde: https://commons.wikimedia.org/wiki/File:RNN.png
# RNN kerroksessa käydään toistorakenteen avulla läpi käsiteltävä sekvenssi askeleittain.
# Samalla pidetään yllä "piiloitettua tilaa", jossa pidetään yllä informaatiota aikaisempien askeleiden tiedoista.

# Kun täysin yhdistetyssä kerroksessa on kaksi painoarvoa: yksi sisääntulolle ja yksi biassille,
# RNN kerroksissa on kolme painoarvoa: yksi sisääntulolle, yksi biasille ja yksi piiloitetulle tilalle.

# Ongelma RNN malleissa on se, että ne unohtavat pitkällä aikavälillä tietoa.
# Tämän takia nykyään yleinen käytäntö on suosia LSTM soluja kuin RNN rakenteita parempien tulosten saavuttamiseksi.

#%%
# RNN Kerassissa
# Luodaan esimerkki, jossa meillä on 5 mittausta, missä mittausarvo alkaa ykkösellä ja nousee aina yhdellä.
import tensorflow as tf
import numpy as np

data = np.arange(1,6) # [1,2,3,4,5]
# data tulee olla kolme dimensiota, eli muokataan se muotoon (1,5,1)
data = data.reshape((1,5,1))
print("shape ", (data.shape))
input_layer = tf.keras.Input(shape=(5,1))
rnn = tf.keras.layers.SimpleRNN(1)(input_layer)
rnn_model = tf.keras.Model(inputs=input_layer,
                           outputs=rnn)
prediction = rnn_model.predict(data)
print(prediction)
# [[-0.9999585]]

#%%
# Oletuksena RNN kerros palauttaa viimeisen piiloitetun tilan ulostulon.
# Tämä riittää, jos RNN kerroksen ulostulo syötetään suoraan esim. Dense kerrokselle.
# Määrittämällä kerrokseen parametri "return_sequences = True", saadaan jokaiselle sisätulon arvolle vastaava piiloitettu tila
input_layer = tf.keras.Input(shape=(5,1))
rnn = tf.keras.layers.SimpleRNN(1,return_sequences=True)(input_layer)
rnn_model = tf.keras.Model(inputs=input_layer,
                           outputs=rnn)
prediction = rnn_model.predict(data)
print(prediction)
#[[[0.9146411 ]
  #[0.97554   ]
  #[0.9987538 ]
  #[0.99994165]
  #[0.9999975 ]]]
  
#%%
# Jos RNN kerroksen perään haluaa määrittää toisen RNN kerroksen,
# tulee edeltävään RNN kerrokseen määrittää return_sequences parametri todeksi.
# Muuten seuraava RNN kerros ei saisi sisääntulona sekvenssiä edellisen kerroksen piiloitetuista tiloista,
# vaan vain viimeisen piiloitetun tilan.
input_layer = tf.keras.Input(shape=(5,1))
rnn = tf.keras.layers.SimpleRNN(1,return_sequences=True)(input_layer)
rnn2 = tf.keras.layers.SimpleRNN(1)(rnn)
rnn_model = tf.keras.Model(inputs=input_layer,
                           outputs=rnn2)
prediction = rnn_model.predict(data)
print(prediction)
# [[0.18753971]]

#%%

# LTSM

# Lähde: https://commons.wikimedia.org/wiki/File:The_LSTM_cell.png

# LSTM solu eroaa pelkistä takaisinkytkeytyvistä kerroksista niin,
# että sen rakenne on monimutkaisempi ja sisältää enemmän tietoa.
# LSTM:ssä on tieto solun sisäisestä tilasta,
# joka välitetään eteenpäin muille LSTM solun sisäisiin tiloihin.
# Tämän lisäksi LSTM sisältää takaisinkytkeytyvien verkkojen tapaan tiedon piiloitetusta tilasta.

#Sisäinen tila on kuvassa ylin vaakasuora viiva, johon tulee sisään Ct−1 ja ulos Ct
# Sisäistä tilaa muokataan kolmella portilla, jotka ovat:

# Forget - portti: Mitä solun sisäisestä tilasta pidetään ennallaan ja mitkä "unohdetaan" eli poistetaan kokonaan.
# Input - portti: Mitä uutta tietoa sisäiseen tilaan syötetään
# Output - portti: Mitä tietoa solusta annetaan eteenpäin.
    # Voidaan ajatella, että tässä portissa luodaan 'suodatettu' versio solun sisäisestä tilasta.

# Portit hyödyntävät piiloitettua tilaa ht−1  ja sisääntuloa xt
# Niin kuin kuvasta nähdään,
    # forget - portti hyödyntää sigmoid - aktivaatiofunktiota,
    # input - portti sigmoid ja tanh - aktivaatiofunktioita.
    
#%%
# LTSM Kerasissa

lstm = tf.keras.layers.LSTM(1)(input_layer)
model = tf.keras.Model(inputs=input_layer,
                       outputs=lstm)
# data tulee olla kolme dimensiota, eli muokataan se muotoon (1,5,1)
data = data.reshape((1,5,1))
prediction = model.predict(data)
print(prediction)
# [[-0.03050232]]

# Syöttämällä mallille 3D muotoista dataa, saatiin LSTM:n ulostulo,
# joka aikaisemmassa kuvassa on piiloitettu tila eli ht , yhdelle sisääntulosekvenssille.

#%%
# Jos halutaan saada LSTM:n piiloitettu tila jokaiselle sisääntulon arvolle,
# voidaan käyttää 'return_sequences' parametriä.
lstm_return_sequences = tf.keras.layers.LSTM(1,return_sequences=True)(input_layer)
model2 = tf.keras.Model(inputs=input_layer,
                        outputs=lstm_return_sequences)
prediction = model2.predict(data)
print(prediction)

'''
[[[0.07039573]
  [0.07901163]
  [0.04846271]
  [0.02332033]
  [0.01009175]]]
'''

#%%
# Jos LSTM kerroksia laitetaan peräkkäin, tulee käyttää 'return_sequences' parametriä.
# Muuten seuraava LSTM kerros saisi syötteenä vain yksittäisen arvon eikä koko sekvenssiä.
lstm1 = tf.keras.layers.LSTM(1,return_sequences=True)(input_layer)
lstm2 = tf.keras.layers.LSTM(1)(lstm1)
model3 = tf.keras.Model(inputs=input_layer,
                        outputs=lstm2)
prediction = model3.predict(data)
print(prediction)
# [[-0.25583985]]

#%%
# LSTM soluilla on ulostulon eli piiloitetun tilan lisäksi myös sisäinen tila.
# Sisäistä tilaa voi käyttää mm. alustaakseen toisen LSTM solun tilan samanlaiseksi kuin sen,
# minkä tila kopioidaan. Sisäisen tilan voi saada antamalla LSTM luokalle parametrin 'return_state'.
lstm3, lstm_hidden_state, lstm_internal_state = tf.keras.layers.LSTM(1, return_state=True)(input_layer)
model4 = tf.keras.Model(inputs=input_layer,
                        outputs=[lstm3,lstm_hidden_state,lstm_internal_state])
prediction = model4.predict(data)
print(prediction)
'''
[
 array([[-0.02507074]], dtype=float32),
 array([[-0.02507074]], dtype=float32),
 array([[-2.578528]], dtype=float32)
 ]
'''
# Tulostuksen viimeinen rivi on tässä tapauksessa siis LSTM:n sisäinen tila.

#%%
# Bidirectional RNN

# Tavallisessa RNN mallissa vain edeltävät tilat vaikuttavat nykyiseen tilaan.
# Bidirectional RNN mallissa myös tulevat tilat voivat takautuvasti vaikuttaa nykyiseen ja edeltäviin tiloihin.
# Näin tieto virtaa verkossa myös takaisinpäin.

# Lähde https://commons.wikimedia.org/wiki/File:Structural_diagrams_of_unidirectional_and_bidirectional_recurrent_neural_networks.png

# Niin kuin yllä olevasta kuvasta nähdään,
# kaksisuuntainen RNN saadaan kopioimalla yksisuuntainen recurrent - kerros,
# jolloin saadaan kaksi kerrosta rinnakkain.
# Toiselle kerrokselle syötetään sisääntulosekvenssi oikeassa järjestyksessä ja toiselle vastakkaisessa järjestyksessä.
# Näin kerrokset oppivat sekvenssin erilailla ja saavat erinlaiset painoarvot.

# Kerassissa RNN tai LSTM - kerroksen voi tehdä kaksisuuntaiseksi laittamalla kerros Bidirectional - wrapperin sisään.

# Luodaan LSTM - kerros
lstm = tf.keras.layers.LSTM(20,return_sequences=True)
# Laitetaan luotu lstm - kerros Bidirectional kutsun sisään
bidirectional_lstm = tf.keras.layers.Bidirectional(lstm)
# Voidaan luoda myös yhdellä rivillä:
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True))

#%%
# RNN mallin käyttö tulevien arvojen ennustamiseen

# Luodaan esimerkkinä sinifunktiolla arvoja, johon on lisätty kohinaa.
# Luodaan neuroverkko tulevien sinifunktion arvojen ennustamiseen.
# Verrataan Dense - neuroverkon ja RNN - neuroverkon tuloksia.
import matplotlib.pyplot as plt
import numpy as np

sin_range = np.arange(0,101,0.1) 
sin_function = np.sin(sin_range) + np.random.normal(0,0.4,sin_range.shape)
plt.plot(sin_function)
plt.show()

#%%
# Datan muokkaus valvottu oppiminen - muotoon

# Aikaisemmassa esimerkissä sinifunktion tulevaisuuden arvoja tulisi ennustaa historiatietojen perusteella.
# Jotta koneoppimisen malleja voitaisiin opettaa, tulee luoda sekvenssejä,
# jossa on esim. ajanhetkessä t0 sinifunktion viimeisimmät 10 arvoa eli t−1, t−2, t−3, ..., t−10 ominaisuuksina
# ja ennustettava arvo olisi tällöin t0.
# Vastaavasti seuraava sekvenssi ajanhetkessä t1
# viimeisimmät 10 arvoa olisivat t0, t−1, t−2, ..., t−9
# ja ennustettava arvo olisi t1.

print("Ensimmäinen sekvenssi")
print("Ominaisuudet:")
print(sin_function[:10])
print("Ennustettava arvo:")
print(sin_function[10], '\n')
print("Toinen sekvenssi")
print("Ominaisuudet:")
print(sin_function[1:11])
print("Ennustettava arvo:")
print(sin_function[11])

'''
Ensimmäinen sekvenssi
Ominaisuudet:
[-0.2020437  -0.47912403  0.20207498  0.64339335  1.31344293 -0.16099651
  0.3667811   0.72998234  0.53588253  0.34315657]
Ennustettava arvo:
1.5130252426304693 

Toinen sekvenssi
Ominaisuudet:
[-0.47912403  0.20207498  0.64339335  1.31344293 -0.16099651  0.3667811
  0.72998234  0.53588253  0.34315657  1.51302524]
Ennustettava arvo:
1.4133829060604166
'''

#%%
# Luodaan ylläolevan manuaalisen esimerkin mukaisesti toistorakenne käymään koko datasetti läpi.
# Laitetaan time_steps pituinen sekvenssi featureita ja sekvenssin jälkeinen arvo labeliksi.
# Eli jos nykyinen ajanhetki on t1, niin sekvenssi sisältää askeleet t1-t10 ja label on t11.
time_steps = 10
features, labels = [],[]

for i in range(0,len(sin_function) - time_steps):
    feature = sin_function[i:(i+time_steps)]
    features.append(feature)
    label = sin_function[i+time_steps]
    labels.append(label)
    
# muutetaan listat numpy - taulukoiksi
features, labels = np.array(features), np.array(labels)
print(features[0], labels[0])
print(features.shape,labels.shape)
'''
[-0.2020437  -0.47912403  0.20207498  0.64339335  1.31344293 -0.16099651
  0.3667811   0.72998234  0.53588253  0.34315657] 1.5130252426304693
(1000, 10) (1000,)
'''

#%%
# Jako koulutus- ja testidataan
train_X, train_y = features[:len(features) // 2], labels[:len(features) // 2]
test_X, test_y = features[len(features) // 2:], labels[len(features) // 2:]

print(train_X.shape,train_y.shape)
print(test_X.shape,test_y.shape)
'''
(500, 10) (500,)
(500, 10) (500,)
'''

#%%
# Seuraavaksi luomme neuroverkon.
# Vertailun vuoksi luodaan ensin täysin yhdistetty neuroverkko, jossa on kaksi piiloitettua kerrosta.
import tensorflow as tf
fcnn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1,activation='linear')
])

fcnn_model.compile(optimizer='adam',
                   loss='mse',
                   metrics=['mean_squared_error'])

fcnn_model.fit(train_X,train_y,epochs=10,verbose=0)
test_results = fcnn_model.evaluate(test_X,test_y,verbose=0)
predictions = fcnn_model.predict(test_X)
print(f"Test loss {test_results[0]}")
'''
Test loss 0.2503667175769806
'''

#%%
# Piirretään ennustukset ja oikeat arvot viivakaavioina

fig, ax = plt.subplots()
fig.set_size_inches(18.5,10.5)
ax.plot(sin_function[len(sin_function) // 2:],label='Test data')
ax.plot(predictions,label='Predictions')
plt.legend()
plt.show()

#%%

# LTSM neuroverkon luonti Kerassilla

# sisääntulo pitää muotoilla 3D muotoon eli (samples, time_steps, features)
# Tässä tapauksissa ominaisuuksia on vain yksi kappale ja time_steps on 10 eli kymmenen viimeisintä arvoa.
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X =  test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
print(train_X.shape)
'''
(500, 10, 1)
'''
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1,activation='linear')
])
lstm_model.compile(optimizer='adam',
                   loss='mse',
                   metrics=['mean_squared_error'])
lstm_model.fit(train_X,train_y,epochs=10,verbose=0)
test_results_lstm = lstm_model.evaluate(test_X,test_y,verbose=0)
predictions_lstm = lstm_model.predict(test_X)
print(f"Test loss {test_results_lstm[0]}")

'''
Test loss 0.22109460830688477
'''

#%%
fig, ax = plt.subplots()
fig.set_size_inches(18.5,10.5)
ax.plot(sin_function[len(sin_function) // 2:],label='Test data')
ax.plot(predictions_lstm,label='Predictions')
plt.legend()
plt.show()

# Huomataan kuvaajista, että pelkkä täysin yhdistetty neuroverkkomallin ennustuksissa on paljon kohinaa,
# kun LSTM verkko on oppinut suurin piirtein aaltomaisen muodon, eikä kohinaa esiinny niin paljon.
#%%

# RNN toiminta https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# LSTM toiminta http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# Aikasarjaesimerkki https://towardsdatascience.com/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python-6ceee9c6c651