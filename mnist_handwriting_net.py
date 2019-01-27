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

    def backward(self, parameter, chain, output, input):
        grad = np.outer(chain, input)
        chain = np.matmul(chain, parameter)
        return (chain, grad)

    def initialParameter(self):
        return (1.0/math.sqrt(self._outputs*self._inputs))*(0.98*np.random.rand(self._outputs, self._inputs) + 0.01)

class LogisticFunc:
    def __init__(self, outputs: int):
        self._outputs = outputs
    
    def forward(self, parameter, input):
        return scipy.special.expit(input)

    def backward(self, parameter, chain, output, input):
        grad = None
        chain = chain*output*(1-output)
        return (chain, grad)
    
    def initialParameter(self):
        return None


class NeuralNet:

    def __init__(self, nodecounts):
        self.nodecounts = nodecounts

        self._layers = []
        for i in range(0, len(self.nodecounts)-1):
            inputs = self.nodecounts[i]
            outputs = self.nodecounts[i+1]
            self._layers.append(DenseSum(outputs, inputs))
            self._layers.append(LogisticFunc(outputs))

        self._parameters = [layer.initialParameter() for layer in self._layers]


    def evaluate(self, input):
        assert len(input) == self.nodecounts[0]

        x = input
        for (layer, parameter) in zip(self._layers, self._parameters):
            x = layer.forward(parameter, x)

        return x


    def train(self, input, target):
        assert len(input) == self.nodecounts[0]
        assert len(target) == self.nodecounts[-1]

        # Forward propagation to get evaluated values in the net
        values = [input] + [None]*len(self._layers)
        for i in range(0,len(self._layers)):
            values[i+1] = self._layers[i].forward(parameter=self._parameters[i],
                                                    input=values[i])

        # Back propagate the gradient chain
        grads = [None]*len(self._layers)
        chain = (values[-1] - target)
        for i in range(len(self._layers)-1,-1,-1):
            (chain, grads[i]) = self._layers[i].backward(parameter=self._parameters[i],
                                                            chain=chain,
                                                            output=values[i+1],
                                                            input=values[i])

        # Gradient step
        assert(len(grads) == len(self._parameters))
        for (parameter, grad) in zip(self._parameters, grads):
            if parameter is not None:
                parameter -= 0.1*grad


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
