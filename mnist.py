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

def costToOutput(weight, input, bias, desired):
    z = calculateZ(weight, input, bias)
    return sigmoidPrime(z) * costPrime(desired, actual(z))

def costToWeight(weight, input, bias, desired):
    return input * costToOutput(weight, input, bias, desired)

def costToInput(weight, input, bias, desired):
    return weight * costToOutput(weight, input, bias, desired)

def costToBias(weight, input, bias, desired):
    return costToOutput(weight, input, bias, desired)

class NeuralNet:
    def __init__(self, inputSize, outputSize, weightSizes):

        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weightSizes = weightSizes

        self.maxSize = max(self.inputSize, self.outputSize, *self.weightSizes)

        numLayers = len(self.weightSizes) + 1

        normalize = lambda x: x * 2 - 1

        self.inputVec = np.zeros((self.maxSize))

        self.weights = normalize(np.random.rand(numLayers, self.maxSize, self.maxSize))
        self.biases = normalize(np.random.rand(numLayers, self.maxSize))

        self.zs = self.biases.copy()
        self.nodes = sigmoid(self.zs)



    def analyze(self, inputVec):
        self.inputVec = np.pad(inputVec, (0, self.maxSize - len(inputVec)))

        self.forward_prop()

    def mnistPredict(self, inputVec):
        self.analyze(inputVec)

        outputs = self.nodes[-1][:self.outputSize]

        return np.argmax(outputs)

    def forward_prop(self):
        for layerNum in range(len(self.weights)):

            if layerNum == 0:
                prevLayer = self.inputVec
            else:
                prevLayer = self.nodes[layerNum - 1]

            prevLayerMatrix = np.full((self.maxSize, self.maxSize), prevLayer)

            sumWeights = np.diag(prevLayerMatrix, self.weights[layerNum])
            
            self.zs[layerNum] = calculateZ(sumWeights, self.biases[layerNum])
            self.nodes[layerNum] = sigmoid(self.zs[layerNum])

net = NeuralNet(4, 2, [3])