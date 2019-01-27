import numpy as np

class SquaredL2ErrorMeasure:
    def __init__(self, inputs: int):
        self._inputs = inputs

    def forward(self, input, target):
        r = input - target
        return np.sum(r*r)
    
    def backward(self, input, target):
        return 2*(input - target)
