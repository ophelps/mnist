import numpy as np
import random

#desired output is y
#actual output is A
#input is I -> same as a^(L-1)


def calculateZ(weight, input, bias):
    return (weight*input)+bias

def actual(weight, input, bias):
    return sig(z(weight, input, bias))

def cost(desired, actual):
    return (desired - actual)**2

def costPrime(desired, actual):
    return 2 * (desired - actual)

def sig(x):
    return 1/(1 + np.exp(-x))

def sigPrime(x):
    u = np.exp(-x)
    return u/((1+u)**2)

#make more efficient later

def costToOutput(weight, input, bias, desired):
    z = calculateZ(weight, input, bias)
    return sigPrime(z) * costPrime(desired, actual(z))

def costToWeight(weight, input, bias, desired):
    return input * costToOutput(weight, input, bias, desired)

def costToInput(weight, input, bias, desired):
    return weight * costToOutput(weight, input, bias, desired)

def costToBias(weight, input, bias, desired):
    return costToOutput(weight, input, bias, desired)

def costToBias(z, desired)

class InputNode:
    def __init__(self, val):
        self.val = val

    def __str(self):
        print(self.val)

class Node:
    def __init__(self, nodeTree):

        self.weights = []
        for previousNode in range(len(nodeTree[-1])):
            self.weights.append(random.random() * 2 - 1)
        
        self.bias = random.random() * 2 - 1

def createNodeTree(inputSize):
    inputLayer = []
    for i in range(inputSize):
        inputLayer.append(InputNode(0))
    
    nodeTree = [inputLayer]
    return nodeTree

def addLayer(nodeTree, layerSize):
    currLayer = []

    for i in range(layerSize):
        inputLayer.append(Node(nodeTree))

z = calculateZ(weight, input, bias)