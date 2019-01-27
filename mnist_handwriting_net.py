# data https://pjreddie.com/projects/mnist-in-csv/

import layers
import measures

import math
import numpy as np
import matplotlib.pyplot as plt
import csv



class NeuralNet:

    def __init__(self, nodecounts):
        self.nodecounts = nodecounts

        self._layers = []
        self._errorMeasure = measures.SquaredL2ErrorMeasure(nodecounts[-1])
        for i in range(0, len(self.nodecounts)-1):
            inputs = self.nodecounts[i]
            outputs = self.nodecounts[i+1]
            self._layers.append(layers.DenseSum(outputs, inputs))
            self._layers.append(layers.LogisticFunc(outputs))

        self._parameters = [layer.initialParameter() for layer in self._layers]


    def evaluate(self, input, target=None):
        assert len(input) == self.nodecounts[0]

        x = input
        for (layer, parameter) in zip(self._layers, self._parameters):
            x = layer.forward(parameter, x)
        if target is None:
            return x
        else:
            return (x, self._errorMeasure.forward(x, target))


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
        chain = self._errorMeasure.backward(input=values[-1],
                                            target=target)
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
    (x,e) = net.evaluate(line[1],line[2])

    i = np.argmax(x)

    if(i == line[0]):
        correct = correct+1

    r = x - line[2]
    
    print("error=%f %d %d" % (e, i, line[0]))
    #print("f=%s" % ["%.2f" % z for z in x])
    #print("t=%s" % ["%.2f" % z for z in line[2]])
    #print("e=%s" % ["%.2f" % z for z in r])

print("correct=%d/%d" % (correct, len(test)))

#plt.imshow(train[0][1].reshape((28,28)))
#plt.show()
