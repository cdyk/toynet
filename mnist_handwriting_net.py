# data https://pjreddie.com/projects/mnist-in-csv/

import network
import trainers

import math
import numpy as np
import matplotlib.pyplot as plt
import csv

def readdata(path, maxRows):
    inputs = []
    targets = []
    solutions = []
    print("Reading %s..." % path)
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            inputs.append((0.99/258.0)*np.asfarray(row[1:]) + 0.01)
            targets.append(np.asfarray([0.99 if x==int(row[0]) else 0.01 for x in range(0,10)]))
            solutions.append(int(row[0]))
            if maxRows <= len(inputs):
                break
    print("...done")
    return (inputs, targets, solutions)

train = readdata("mnist_train.csv", 1000)
test = readdata("mnist_test.csv", 100)

net = network.NeuralNet([784, 100, 10])
trainers.stochasticGradientDescent(net, train[0], train[1], 10)

correct = 0

for (input,target,solution) in zip(test[0], test[1],test[2]):
    (x,e) = net.evaluate(input, target)

    i = np.argmax(x)

    if(i == solution):
        correct = correct+1
    
    #print("error=%f %d %d" % (e, i, solution))
    #print("f=%s" % ["%.2f" % z for z in x])
    #print("t=%s" % ["%.2f" % z for z in line[2]])
    #print("e=%s" % ["%.2f" % z for z in r])

print("correct=%d/%d" % (correct, len(test[0])))

#plt.imshow(train[0][1].reshape((28,28)))
#plt.show()
