import layers
import measures

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

    def gradient(self, input, target):
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
        return grads

    def adjustWeights(self, adjustments):
        assert(len(adjustments) == len(self._parameters))

        for (parameter, adjustment) in zip(self._parameters, adjustments):
            if parameter is not None:
                parameter -= adjustment