import math
import numpy as np
import scipy.special


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
