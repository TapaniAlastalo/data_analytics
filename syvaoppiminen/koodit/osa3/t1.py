# Tehtävän toteutus

import numpy as np

# Lataa clickbait ja ei-clickbait otsikot clickbait_data.txt ja non_clickbait_data.txt tiedostoista data - kansiosta
file1 = open('c:/data/clickbait_data.txt', 'r', encoding='UTF-8')
lines1 = file1.readlines()
np1 = np.array(lines1)
print(np1.shape)

file2 = open('c:/data/non_clickbait_data.txt', 'r', encoding='UTF-8')
lines2 = file2.readlines()
np2 = np.array(lines2)
print(np2.shape)

'''
np1 = np1.reshape((np1.size,1))
o = np.ones((np1.size,1))
np1 = np.append(np1, o, axis=1)
print(np1.shape)
#print(np1)


np2 = np2.reshape((np2.size,1))
z = np.zeros((np2.size,1))
np2 = np.append(np2, z, axis=1)
print(np2.shape)
#print(np2)
'''


'''
lines2 = []
for line in file2:
    print(line)
    lines2.append(line)

np2 = np.array(lines2)
print(np2.shape)
'''

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

# sisääntulo pitää muotoilla 3D muotoon eli (samples, time_steps, features)
# Tässä tapauksissa ominaisuuksia on vain yksi kappale ja time_steps on 10 eli kymmenen viimeisintä arvoa.

train_X = train_X.reshape(train_X.shape[0], 1, 1)
test_X =  test_X.reshape(test_X.shape[0], 1, 1)

#train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
#test_X =  test_X.reshape(test_X.shape[0], test_X.shape[1], 1)


print(train_X.shape,train_y.shape)
print(test_X.shape,test_y.shape)

#%%
# Luo RNN - malli, joka ennustaa, onko otsikko clickbait vai ei
import tensorflow as tf
print("model")

#features = features.reshape((1,58001,1))
#print("shape ", (features.shape))


# data tulee olla kolme dimensiota, eli muokataan se muotoon (1,5,1)
input_layer = tf.keras.Input(shape=(5,1))#train_X.shape[0],1))
rnn = tf.keras.layers.SimpleRNN(1,return_sequences=True)(input_layer)
rnn2 = tf.keras.layers.SimpleRNN(1)(rnn)
model = tf.keras.Model(inputs=input_layer,
                           outputs=rnn2)

print(model.summary())

#print(model.predict(train_X))


''''
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1,activation='linear')
])
'''

#%%
# Tulosta mallin tarkkuus evaluate - funktiolla
# Seuraavaksi luomme neuroverkon.
# Vertailun vuoksi luodaan ensin täysin yhdistetty neuroverkko, jossa on kaksi piiloitettua kerrosta.
print("compile")
#model.compile(optimizer='adam',
#                   loss='categorical_crossentropy', #'mse'
#                   metrics=['accuracy']) #['mean_squared_error'])

model.compile(optimizer='adam',
                   loss='mse',
                   metrics=['mean_squared_error'])

print("train")
model.fit(train_X, train_y, epochs=3, verbose=0)
print("evaluate")
test_results = model.evaluate(test_X, test_y, verbose=0)
print("predict")
predictions = model.predict(test_X)
print(f"Test loss {test_results[0]}")

#%%
# Aja "Tehtävän vastaukset" solu

# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. Sijoita results - muuttujaan funktion model.evaluate() tulos.
# Muista määrittää model.compile() - funktioon seurattavaksi suureeksi metrics=['accuracy'], jotta näät, kuinka suuri osa neuroverkon ennustuksista on oikein.
print(f"Test Loss:{results[0]} Test Accuracy:{results[1]*100}%")