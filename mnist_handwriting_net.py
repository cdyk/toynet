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
        #                     = sigma_2  1 x n3 matrix
       
        sigma[1] = (outs[2]-target)*outs[2]*(1-outs[2])

        # J(s_2;w_ij) = n3 x n2 matrix element w_ij nonzero
        # -> sigma_2 J(s_2;w_ij) = ( 0 ... sigma_3_j o_2_i ... 0)
        #
        #          +-                                         -+
        #          | sigma_3_1 o_2_1   ...   sigma_3_n3 o_2_1  |
        # delta =  |      ..                        ..         |
        #          | sigma_3_1 o_2_n2  ...   sigma_3_n3 o_2_n2 |
        #          +-                                         -+
        delta[1] = np.outer(sigma[1], outs[1])

        # calculate sigma 1 from sigma 2,
        # dE;dw_ij = G(e;o_3) J(o_3;w_ij)
        #          = G(e;o_3) J(o_3;s_2) J(s_2;w_ij)
        #          = G(e;o_3) J(o_3;s_2) J(s_2;o_2) J(o_2;w_ij)
        #          = G(e;o_3) J(o_3;s_2) J(s_2;o_2) J(o_2;s_1) J(s_1;w_ij)
        #           +---- sigma_2 -----+
        #           +----------- sigma_1 --------------------+
        # sigma_1 = sigma_2 J(s_2;o_2) J(o_2;s_1)
        #
        # 

        #   dE;dw_ij = G(e;o_3) J(o_3;s_2) J(s_2;o_2) J(o_2;s_1) J(s_1;w_ij)    <- use if w_ij in s_1
        #             = sigma_2 J(s_2;o_2) J(o_2;s_1) J(s_1;w_ij)
        #              +------- sigma_1 -------------+
        #    sigma_1 = sigma_2 J(s_2;o_2) J(o_2;s_1)
        #
        #
        #                 +-                                -+
        #                 | ds_2_1;do_2_1 ... ds_2_1;do_2_n2 |   s_2_i = < ( w_i1 ... w_im), (o_2_1, ... o_2_m) >
        #    J(s_2;o_2) = |      ...               ...       |
        #                 | ds_2_m;do_2_1 ... ds_2_m;do_2_n2 |   ds_2_i;do_2_j = w_ij 
        #                 +-                                -+
        #                 +-              -+
        #                 | w_11  ... w_1m |
        #               = | ...       ...  | = W
        #                 | w_n1  ... w_nm |
        #                 +-              -+
        #
        #    J(o_2;s_1) = I( o_2_1;s_1_1 ... o_2_)

        sigma[0] = np.matmul(sigma[1], self.weights[1])*outs[1]*(1-outs[1])
        delta[0] = np.outer(sigma[0], outs[0])



        #a = self.evaluate(input)-target         # before any adjustment
        self.weights[1] -= 0.1*delta[1]
        #b = self.evaluate(input)-target         # after adjustment of last layer
        self.weights[0] -= 0.1*delta[0]
        #c = self.evaluate(input)-target         # after adjustment of last layer

        #print("%f -> %f -> %f" % (np.sum(a*a), np.sum(b*b), np.sum(c*c)))



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
