import random
import network


def stochasticGradientDescent(net: network.NeuralNet, inputs, targets, epochs : int):
    assert(len(inputs) == len(targets))

    indices = list(range(0, len(inputs)))
    for i in range(0,epochs):
        print("Epoch %d" % i)

        random.shuffle(indices)

        for i in indices:
            grads = net.gradient(inputs[i], targets[i])

            for grad in grads:
                if grad is not None:
                    grad *= 0.1

            net.adjustWeights(grads)
