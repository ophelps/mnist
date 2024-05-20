import numpy as np
import random

#desired output is y
#actual output is A
#input is I -> same as a^(L-1)


def calculateZ(weight, input, bias):
    return (weight*input)+bias

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
        maxSize = max(inputSize, outputSize, *weightSizes)

        numLayers = len(weightSizes) + 2

        normalize = lambda x: x * 2 - 1

        self.weights = normalize(np.random.rand(numLayers - 1, maxSize, maxSize))
        self.biases = normalize(np.random.rand(numLayers, maxSize))

        self.zs = self.biases.copy()

        self.nodes = sigmoid(self.zs)


net = NeuralNet(4, 2, [3])