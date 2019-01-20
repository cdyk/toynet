# data https://pjreddie.com/projects/mnist-in-csv/

import math
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import csv

class NeuralNet:

    def __init__(self, nodecounts):
        self.nodecounts = nodecounts
        self.weights = [None]*(len(self.nodecounts)-1)

        for i in range(0, len(self.nodecounts)-1):
            cols = self.nodecounts[i]
            rows = self.nodecounts[i+1]
            self.weights[i] = (1.0/math.sqrt(cols*rows))*(0.98*np.random.rand(rows, cols) + 0.01)

            rowsums = np.sum(self.weights[i], axis=1)

            print("weights %d %d rowsum avg=%s" % (rows, cols, np.average(rowsums)))

    def evaluate(self, input):
        assert len(input) == self.nodecounts[0]
        x = input
        for weight in self.weights:
            x = weight.dot(x)
            x = scipy.special.expit(x)      # expit is logistic func 1/(1+exp(-x))
        return x

    def train(self, input, target):
        assert len(input) == self.nodecounts[0]
        assert len(target) == self.nodecounts[-1]

        nets = []
        outs = [input]
        sigma = [None]*len(self.weights)
        delta = [None]*len(self.weights)

        for weight in self.weights:
            nets.append(weight.dot(outs[-1]))
            outs.append(scipy.special.expit(nets[-1]))      # expit is logistic func 1/(1+exp(-x))

        # Last layer:
        #
        #     dE;dw_ij = G(e;o_3) J(o_3;w_ij)
        #              = G(e;o_3) J(o_3;s_2) J(s_2;w_ij)
        #               +---- sigma --------+
        #
        #     G(E;o_3) = ( E;o_3_1 ... E;o_3_n3 )
        #              = ( -(t_1 - o_3_1) .. -(t_m - o_3_n3) )    // 1 x n3 matrix
        #     J(o_3;s_2) = I(o_3_1;s_2_1 ... o_3_1;s_2_n3)
        #

        # Calculate sigma for last layer
        #
        # G(E;o_3) J(o_3;s_2) = ( dE;do_3_1 do_3_1;ds_2_1 ... dE;do_3_n3 do_3_1;ds_2_n3)
        #                     = ( (o_3_1 - t_1) phi'(s_2_1) ... (o_3_n3 - t_n3)*phi'(s_2_n3) )
        #                     = ( (o_3_1 - t_1) o_3_1 (1 - o_3_1) ... (o_3_n3 - t_n3) o_3_n3 (1 - o_3_n3) )
        #                     = sigma_3  1 x n3 matrix
       
        sigma[1] = (outs[2]-target)*outs[2]*(1-outs[2])

        # J(s_2;w_ij) = n3 x n2 matrix element w_ij nonzero
        # -> sigma_3 J(s_2;w_ij) = ( 0 ... sigma_3_j o_2_i ... 0)
        #
        #          +-                                         -+
        #          | sigma_3_1 o_2_1   ...   sigma_3_n3 o_2_1  |
        # delta =  |      ..                        ..         |
        #          | sigma_3_1 o_2_n2  ...   sigma_3_n3 o_2_n2 |
        #          +-                                         -+

        delta[1] = np.outer(sigma[1], outs[1])

        a = self.evaluate(input)-target         # before any adjustment
        self.weights[1] -= 0.1*delta[1]
        b = self.evaluate(input)-target         # after adjustment of last layer

        print("%f -> %f" % (np.sum(a*a), np.sum(b*b)))



net = NeuralNet([784, 100, 10])



train=[]
with open("mnist_train.csv") as f:
    for l in f.readlines():
        e = l.split(',')
        train.append((e[0],
                     (0.99/258.0)*np.asfarray(e[1:]) + 0.01,
                     np.asfarray([0.99 if x==e[0] else 0.01 for x in range(0,10)])))
        if 10 < len(train):
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
        print("f=%s" % ["%.2f" % z for z in x])
        print("t=%s" % ["%.2f" % z for z in test[-1][2]])
        print("e=%s" % ["%.2f" % z for z in r])

print("correct=%d/%d" % (correct, len(test)))

#plt.imshow(train[0][1].reshape((28,28)))
#plt.show()
