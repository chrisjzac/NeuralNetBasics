""" Imports """
import numpy as np

class Layer():

    def __init__(self, n_inputs, n_neurons, act=0):
        self.weights = np.random.random((n_neurons, n_inputs))
        self.biases = np.random.randint(0,10,n_neurons)
        self.output = None
        self.act = act

    def sigmoid(self):
        self.output = 1/(1 + np.exp(-float(self.output)))

    def relu(self):
        np.maximum(0, self.output)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights.T) + self.biases
        if self.act == 1:
            self.sigmoid()
        else:
            self.relu()


values = np.random.random((10,3))
layer1 = Layer(3,4)
layer1.forward(inputs=values)
#print(layer1.output)

layer2 = Layer(4,3)
layer2.forward(inputs=layer1.output)
#print(layer2.output)

layer3 = Layer(3,2)
layer3.forward(inputs=layer2.output)
#print(layer3.output)

layer4 = Layer(2,1)
layer4.forward(inputs=layer3.output)
print(layer4.output)