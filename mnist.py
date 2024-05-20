import numpy as np
import random

#desired output is y
#actual output is A
#input is I -> same as a^(L-1)


def calculateZ(weight, bias):
    return weight + bias

def actual(weight, input, bias):
    return sigmoid(calculateZ(weight, input, bias))

def cost(desired, actual):
    return (desired - actual)**2

def costPrime(desired, actual):
    return 2 * (desired - actual)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoidPrime(x):
    u = np.exp(-x)
    return u/((1+u)**2)

#make more efficient later

class NeuralNet:
    def __init__(self, inputSize, outputSize, weightSizes, learning_rate=0.01, epoch_size=100):

        self.epoch_size = epoch_size
        self.learning_rate = learning_rate

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weightSizes = weightSizes

        self.maxSize = max(self.inputSize, self.outputSize, *self.weightSizes)

        self.numLayers = len(self.weightSizes) + 1


        self.inputVec = np.zeros((self.maxSize))

        self.weights = np.random.rand(self.numLayers, self.maxSize, self.maxSize) * 20 - 10
        self.biases = np.random.rand(self.numLayers, self.maxSize) * 20 - 10

        self.zs = self.biases.copy()
        self.nodes = sigmoid(self.zs)

    def __str__(self):
        description = "Neural Net: \n"
        description +="---------------------------\n"
        description +="Weights: " + np.array2string(self.weights) + "\n"
        description +="Biases: " + np.array2string(self.biases) + "\n"
        description +="---------------------------\n"

        return description

    def analyze(self, inputVec):
        self.inputVec = np.pad(inputVec, (0, self.maxSize - len(inputVec)))

        self.forward_prop()

        return self.nodes[-1][:self.outputSize]

    def predict(self, inputVec):
        
        outputs = self.analyze(inputVec)

        return np.argmax(outputs)

    def forward_prop(self):
        #print(self.inputVec)
        #print(self.nodes)
        #print(self.weights)
        #print(self.biases)

        for layerNum in range(len(self.weights)):

            if layerNum == 0:
                prevLayer = self.inputVec
            else:
                prevLayer = self.nodes[layerNum - 1]

            prevLayerMatrix = np.full((self.maxSize, self.maxSize), prevLayer)

            sumWeights = np.diag(np.matmul(prevLayerMatrix, self.weights[layerNum]))
            
            self.zs[layerNum] = calculateZ(sumWeights, self.biases[layerNum])
            self.nodes[layerNum] = sigmoid(self.zs[layerNum])
        
        #print(self.nodes)
        

    def back_prop(self, inputVec, expected):
        self.analyze(inputVec)

        expectedPad = np.pad(expected, (0, self.maxSize - len(expected)))

        #dCost/dOutput
        deltas = costPrime(self.nodes[-1], expectedPad)
        #dOutput/dZ
        deltas *= sigmoidPrime(self.zs[-1])

        weight_changes = []
        bias_changes = []

        for i in range(self.numLayers - 1, -1, -1):
            
            delta_weights = []
            for delta in deltas:
                delta_weights.append(delta * self.nodes[i - 1])

            weight_changes.append(delta_weights)
            bias_changes.append(deltas)

            deltas = np.sum(self.weights[i], axis=0) * deltas
        
        return (weight_changes[::-1], bias_changes[::-1])
    
    def train_epoch(self, inputs, outputs):
        dataI = [*range(len(inputs))]
        random.shuffle(dataI)
        
        total_weight_changes = np.zeros((self.numLayers, self.maxSize, self.maxSize))
        total_bias_changes = np.zeros((self.numLayers, self.maxSize))

        epoch_size = min(self.epoch_size, len(inputs))

        for i in range(epoch_size):
            
            weight_changes, bias_changes = self.back_prop(inputs[dataI[i]], outputs[dataI[i]])

            total_weight_changes += weight_changes
            total_bias_changes += bias_changes
        
        total_weight_changes /= epoch_size
        total_bias_changes /= epoch_size

        total_weight_changes *= self.learning_rate
        total_bias_changes *= self.learning_rate

        self.weights += total_weight_changes
        self.biases += total_bias_changes
    
    def train(self, inputs, outputs, epochs):
        for epoch in range(epochs):
            self.train_epoch(inputs, outputs)

            #print(self)

            print("PREDICTION FOR 0,0: " + str(self.predict([0, 0])))
            print("PREDICTION FOR 0,1: " + str(self.predict([0, 1])))
            print("PREDICTION FOR 1,0: " + str(self.predict([1, 0])))
            print("PREDICTION FOR 1,1: " + str(self.predict([1, 1])))

net = NeuralNet(2, 2, [2], 0.1)

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [[1, 0], [0, 1], [0, 1], [1, 0]]

net.train(inputs, outputs, 10000)

"""
print("PREDICTION FOR 0,0: " + str(net.predict([0, 0])))
print("PREDICTION FOR 0,1: " + str(net.predict([0, 1])))
print("PREDICTION FOR 1,0: " + str(net.predict([1, 0])))
print("PREDICTION FOR 1,1: " + str(net.predict([1, 1])))
"""