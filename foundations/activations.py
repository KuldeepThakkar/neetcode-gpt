import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        out = 1/(1+np.exp(-z))
        # Formula: 1 / (1 + e^(-z))

        # return np.round(your_answer, 5)
        return np.round(out,5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        out= np.maximum(0,z)
        # Formula: max(0, z) element-wise
        return out
