# data https://pjreddie.com/projects/mnist-in-csv/

import math
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import csv

class DenseSum:
    def __init__(self, outputs: int, inputs: int):
        self._outputs = outputs
        self._inputs = inputs

    def forward(self, parameter, input):
        return parameter.dot(input)

    def backward(self, parameter, sigma, output):
        sigma = np.matmul(sigma, parameter)
        delta = np.outer(sigma, output)
        return (sigma, delta)

    def initialParameter(self):
        return (1.0/math.sqrt(self._outputs*self._inputs))*(0.98*np.random.rand(self._outputs, self._inputs) + 0.01)

class LogisticFunc:
    def __init__(self, outputs: int):
        self._outputs = outputs
    
    def forward(self, parameter, input):
        return scipy.special.expit(input)

    def backward(self, parameter, sigma, output):
        sigma = output*(1-output)
        delta = None
        return (sigma, delta)
    
    def initialParameter(self):
        return None


class NeuralNet:

    def __init__(self, nodecounts):
        self.nodecounts = nodecounts
        self.weights = [None]*(len(self.nodecounts)-1)

        self._layers = []
        for i in range(0, len(self.nodecounts)-1):
            inputs = self.nodecounts[i]
            outputs = self.nodecounts[i+1]
            self._layers.append(DenseSum(outputs, inputs))
            self._layers.append(LogisticFunc(outputs))
        self._parameters = [layer.initialParameter() for layer in self._layers]

        for i in range(0, len(self.nodecounts)-1):
            cols = self.nodecounts[i]
            rows = self.nodecounts[i+1]
            self.weights[i] = (1.0/math.sqrt(cols*rows))*(0.98*np.random.rand(rows, cols) + 0.01)

            rowsums = np.sum(self.weights[i], axis=1)

            print("weights %d %d rowsum avg=%s" % (rows, cols, np.average(rowsums)))

    def evaluate(self, input):
        assert len(input) == self.nodecounts[0]
        if False:
            weights = [
                self._parameters[0],
                self._parameters[2]
            ]
            x = input
            for weight in weights:
                x = weight.dot(x)
                x = scipy.special.expit(x)      # expit is logistic func 1/(1+exp(-x))

        if True:
            x = input
            for (layer, parameter) in zip(self._layers, self._parameters):
                x = layer.forward(parameter, x)


        return x

    def train(self, input, target):
        assert len(input) == self.nodecounts[0]
        assert len(target) == self.nodecounts[-1]

        weights = [
            self._parameters[0],
            self._parameters[2]
        ]
        nets = []
        outs = [input]
        sigma = [None]*len(weights)
        delta = [None]*len(weights)
        for weight in weights:
            nets.append(weight.dot(outs[-1]))
            outs.append(scipy.special.expit(nets[-1]))      # expit is logistic func 1/(1+exp(-x))
        sigma[1] = (outs[2]-target)*outs[2]*(1-outs[2])
        delta[1] = np.outer(sigma[1], outs[1])
        sigma[0] = np.matmul(sigma[1], weights[1])*outs[1]*(1-outs[1])
        delta[0] = np.outer(sigma[0], outs[0])

        #a = self.evaluate(input)-target         # before any adjustment
        weights[1] -= 0.1*delta[1]
        #b = self.evaluate(input)-target         # after adjustment of last layer
        weights[0] -= 0.1*delta[0]
        #c = self.evaluate(input)-target         # after adjustment of last layer
        #print("%f -> %f -> %f" % (np.sum(a*a), np.sum(b*b), np.sum(c*c)))

        if True:
            x = input
            forw = []
            for (layer, parameter) in zip(self._layers, self._parameters):
                x = layer.forward(parameter, x)
                forw.append((x, layer, parameter))

            grads = []
            sigma = (forw[-1][0] - target)
            for (output, layer, parameter) in reversed(forw):
                (sigma, delta) = layer.backward(parameter, sigma, output)
                grads.insert(0,delta)

            assert(len(grads) == len(self._parameters))
    #        for (grad, param) in zip(grads, self._parameters):
    #            if param is not None:
    #                param -= 0.1*grad


net = NeuralNet([784, 100, 10])

def readdata(path, maxRows):
    rv = []
    print("Reading %s..." % path)
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            rv.append((int(row[0]),
                      (0.99/258.0)*np.asfarray(row[1:]) + 0.01,
                      np.asfarray([0.99 if x==int(row[0]) else 0.01 for x in range(0,10)])))
            if maxRows < len(rv):
                break
    print("...done")
    return rv


train = readdata("mnist_train.csv", 1000)

for i in range(0,10):
    print("Epoch %d" % i)
    for line in train:
        net.train(line[1], line[2])

correct = 0

test = readdata("mnist_test.csv",10)

for line in test:
    x = net.evaluate(line[1])

    i = np.argmax(x)

    if(i == line[0]):
        correct = correct+1

    r = x - line[2]
    
    #print("error=%f %d %d" % (np.sum(0.5*r*r), i, line[0]))
    #print("f=%s" % ["%.2f" % z for z in x])
    #print("t=%s" % ["%.2f" % z for z in line[2]])
    #print("e=%s" % ["%.2f" % z for z in r])

print("correct=%d/%d" % (correct, len(test)))

#plt.imshow(train[0][1].reshape((28,28)))
#plt.show()
