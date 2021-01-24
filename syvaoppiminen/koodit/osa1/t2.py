import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
from sklearn.datasets import make_blobs

class NeuralNetwork:
    def __init__(self):
        # Alustetaan painoarvot satunaisesti
        self.weights1 = np.random.rand(2,6)
        self.weights2 = np.random.rand(6,1)
        
    # Virhefunktio
    def loss_function(self,true,prediction):
        return 0.5 * (true - prediction)**2
    
    # Virhefunktion derivaatta
    def loss_function_derivative(self,true,prediction):
        return true - prediction
    
    # Sigmoid funktio
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    # Sigmoidin derivaatta
    def sigmoid_derivative(self,x):
        return x * (1 - x)
    
    # Syöte neuroverkon kerrosten läpi
    def forwardpropagation(self,X,y):
        # Määritetään ominaisuudet ja halutut arvot NumPy - taulukoiksi sekä lasketaan kerrosten ulostulot
        self.X = np.array([X])
        self.y = np.array([y])
        self.layer1 = self.sigmoid(np.dot(self.X, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
        self.loss = self.loss_function(self.y, self.output)
    
    # Painoarvojen päivitys backpropagation - menetelmällä
    def backpropagation(self, learning_rate):
        output_error = self.loss_function_derivative(self.y, self.output)
        output_delta = output_error * self.sigmoid_derivative(self.output)
        weights2_adjustment = np.dot(self.layer1.T, output_delta)

        layer1_error = np.dot(output_delta, self.weights2.T)
        layer1_delta = layer1_error * self.sigmoid_derivative(self.layer1)
        weights1_adjustment = np.dot(self.X.T, layer1_delta)
        
        self.weights2 += learning_rate * weights2_adjustment
        self.weights1 += learning_rate * weights1_adjustment


dots, labels = make_blobs(n_samples=40,centers=2,n_features=2,random_state=50)
colors = ['red' if label == 0 else 'blue' for label in labels]
plt.scatter(dots[:,0],dots[:,1],color=colors)

#df = pd.DataFrame({'x': dots[:,0], 'y': dots[:,1], 'labels':labels}) # ei tarvita dataframea
# ominaisuudet
train_x = dots
train_y = labels

# koulutusdatasetti 1000 kertaa neuroverkon läpi
neural_network = NeuralNetwork()
epoch_losses = []
epochs = 1000
for epoch in range(1,epochs+1):
    batch_losses = [] # Lista, johon lisätään yhden kierroksen virhearvot
    for X,y in zip(train_x,train_y):
        neural_network.forwardpropagation(X,y)
        neural_network.backpropagation(learning_rate = 0.01)
        batch_losses.append(neural_network.loss[0])
    epoch_losses.append(np.average(batch_losses)) # Otetaan keskiarvo yhden kierroksen virhearvoista
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss: {neural_network.loss[0]}")


# Kuvaaja virheen muutoksesta koulutuksen aikana
fig, ax = plt.subplots()
ax.plot(epoch_losses)
ax.set_title("Neural Network Loss")
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.show()

# testidata
test_x = [0.1, -5.8]

# testidatan ajaminen
layer1_output = neural_network.sigmoid(np.dot(test_x, neural_network.weights1))
test_output = neural_network.sigmoid(np.dot(layer1_output, neural_network.weights2))
print('Lopputulos:' + str(test_output))
rounded_output = np.round(test_output, 0)
output = 'red' if rounded_output == 0 else 'blue'

# Tehtävän vastaukset. Huom! Älä muokkaa tätä solua, vaan aja se, kun olet suorittanut tehtävän. Sijoita output - muuttujaan neuroverkon ennuste syötölle [0.1,-5.8]
print(output)