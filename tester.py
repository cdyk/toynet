# data https://pjreddie.com/projects/mnist-in-csv/

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import csv

class NeuralNet:

    def __init__(self, nodecounts):
        self.nodecounts = nodecounts
        self.weights = [None]*(len(self.nodecounts)-1)

        for i in range(0, len(self.nodecounts)-1):
            cols = self.nodecounts[i]+1
            rows = self.nodecounts[i+1]
            self.weights[i] = 0.01*np.random.rand(rows, cols) + 0.01
            print("weights %d %d" % (rows, cols))

    def evaluate(self, input):
        assert len(input) == self.nodecounts[0]
        x = input
        for weight in self.weights:
            x = weight.dot(np.append(x,1))  # append 1 for bias
            x = scipy.special.expit(x)      # expit is logistic func 1/(1+exp(-x))
        return x

    def train(self, input, target):
        assert len(input) == self.nodecounts[0]
        assert len(target) == self.nodecounts[-1]

        # phi is logistic func 1/(1+exp(-x))
        # phi'(x) = phi(x)(1-phi(x))


        # out_j = phi( sum_k(w_kj * in_k) )
        #
        # net_j = sum_k w_kj in_k
        #
        # E_j = 0.5(t-out_j)^2
        #
        # dE/dw_ij = dE/dout_j * dout_j/dw_ij
        #          = dE/dout_j * phi'( net_j ) * d(net_j)/dw_ij
        #          = dE/dout_j * phi'( net_j ) * in_i

        # If out_j in last layer,
        #
        # dE/dout_j = (t-out_j)

        # nodes[k][0] - sum of net at k
        # nodes[k][1] - output of activation function at k
        nodes = [(None,np.asfarray(input))]
        for weight in self.weights:
            sum = weight.dot(np.append(nodes[-1][1],1)) 
            out = scipy.special.expit(sum)
            nodes.append((sum,out))


        # partial derivatives up to and including that functions activation func but before net sum
        sigma = [None]*len(self.weights)

        # output layer
        out = nodes[-1][1]
        sigma[-1] = (out - target) * out * (1 - out)

        # Last inner layer
        #
        # If out_j is in inner layer, let net_l be the nodes
        # receinving input from out_j,
        #
        # dE/dout_j = sum_l dE/dnet_l * dnet_l/dout_j
        #           = sum_l dE/dout_l * dout_l/dnet_l * dnet_l/dout_j
        #           = sum_l dE/dout_l * dout_l/dnet_l * w_jl

        # transform input of last activation func to input of last sum (at output of prev layer)
        psi = ((np.transpose(self.weights[-1]).dot(sigma[-1]))[:-1])

        # move through activation function of next to last layer
        out = nodes[-2][1]
        sigma[-2] = psi * out * (1 - out)

        self.weights[-1] -= 0.1 * np.outer(sigma[-1], np.append(nodes[-2][1],1))
        self.weights[-2] -= 0.1 * np.outer(sigma[-2], np.append(nodes[-3][1],1))


net = NeuralNet([784, 100, 10])



train=[]
with open("mnist_train.csv") as f:
    for l in f.readlines():
        e = l.split(',')
        train.append((e[0],
                     (0.99/258.0)*np.asfarray(e[1:]) + 0.01,
                     np.asfarray([0.99 if x==e[0] else 0.01 for x in range(0,10)])))
        if 10000 < len(train):
            break

        net.train(train[-1][1], train[-1][2])

correct = 0

test=[]
with open("mnist_test.csv") as f:
    for l in f.readlines():
        e = l.split(',')
        test.append((int(e[0]),
                     (0.99/258.0)*np.asfarray(e[1:]) + 0.01,
                     np.asfarray([0.99 if x==int(e[0]) else 0.01 for x in range(0,10)])))
        if 10 < len(test):
            break

        x = net.evaluate(test[-1][1])

        i = np.argmax(x)

        if(i ==  test[-1][0]):
            correct = correct+1

        r = x - test[-1][2]
        
        print("error=%f %d %d" % (np.sum(0.5*r*r), i, test[-1][0]))
        #print(x)
        #print(r)

print("correct=%d/%d" % (correct, len(test)))

#plt.imshow(train[0][1].reshape((28,28)))
#plt.show()
