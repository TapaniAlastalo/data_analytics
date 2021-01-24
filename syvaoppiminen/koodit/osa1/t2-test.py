import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from IPython.display import Image
from matplotlib.ticker import FormatStrFormatter

# Funktio hienompia graafeja varten
def create_fig(x,y):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Muokataan koordinaatisto näyttämään origo keskellä alhaalla
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.plot(x,y)
    plt.show()

dots, labels = make_blobs(n_samples=40,centers=2,n_features=2,random_state=50)
colors = ['red' if label == 0 else 'blue' for label in labels]
#plt.scatter(dots[:,0],dots[:,1],color=colors)


# ominaisuudet
train_x = dots
train_y = labels


b1 = 0.2
b2 = 0.6
w = np.random.rand(train_x.shape[1], train_x.shape[0])
wz = np.random.rand(w.shape[1],1)
print(w.shape)
print(wz.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y1 = sigmoid(np.dot(train_x, w) + b1)
y2 = sigmoid(np.dot(y1,wz) + b2) 
print(y1)
print(y2)

weights1 = np.random.rand(2,6)
weights2 = np.random.rand(6,1)

# Lasketaan pistetulo sisääntulolle ja painoarvoille
layer1 = np.dot(train_x[0], weights1)
layer1 = np.array([layer1])
print(f"Kerroksen arvot ennen sigmoid - aktivaatiofunktiota: {layer1}")
# Syötetään pistetulon tulos sigmoid - funktiolle
layer1 = sigmoid(layer1)
print(f"Kerroksen arvot sigmoid - aktivaaitofunktion jälkeen: {layer1}")

# ulostulon ja weights2 painoarvojen pistetulo
neural_network_output = sigmoid(np.dot(layer1,weights2))
print(neural_network_output)

# yhdistettynä
def forwardpropagation(X, weights1, weights2):
    layer1 = sigmoid(np.dot(X, weights1))
    output = sigmoid(np.dot(layer1, weights2))

error = 0.5 * (train_y[0] - neural_network_output)**2
print(error)

def loss_function_derivative(true,prediction):
    return true - prediction
def sigmoid_derivative(x):
    return x * (1 - x)


learning_rate = 0.1 # oppimisnopeus
y_sample = np.array([train_y[0]]) # muutetaan ensimmäinen arvo halutuista arvoista taulukkomuotoon
output_error = loss_function_derivative(y_sample,neural_network_output) # Ulostulon virhe, laskettuna virhefunktion derivaatalla
output_delta = output_error * sigmoid_derivative(neural_network_output) # Ulostulon delta summa
weights2_adjustment = np.dot(layer1.T,output_delta) # Lasketaan ulostulon virheen ja deltan avulla, kuinka paljon painoarvoja muokataan
weights2 += learning_rate * weights2_adjustment # Kerrotaan painoarvon muutos oppimisnopeudella ja lisätään/vähennetään se weights2 - muuttujasta

layer1_error = np.dot(output_delta, weights2.T) # Piiloitetun kerroksen virhe
layer1_delta = layer1_error * sigmoid_derivative(layer1) # Piiloitetun kerroksen delta summa
x_sample = np.array([train_x[0]]) # muutetaan ensimmäinen rivi ominaisuuksista taulukkomuotoon
weights1_adjustment = np.dot(x_sample.T,layer1_delta) # Lasketaan virheen ja delta summan avulla, kuinka paljon painoarvoja muokataan
weights1 += learning_rate * weights1_adjustment # Kerrotaan painoarvon muutos oppimisnopeudella ja lisätään/vähennetään se weights1 - muuttujasta

print(weights1,'\n\n',weights2)

