import network

def stochasticGradientDescent(net: network.NeuralNet, inputs, targets, epochs : int):
    for i in range(0,epochs):
        print("Epoch %d" % i)

        for (input,target) in zip(inputs, targets):
            grads = net.gradient(input, target)
            for grad in grads:
                if grad is not None:
                    grad *= 0.1

            net.adjustWeights(grads)