import numpy as np

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
    return 2*(desired - actual)

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
