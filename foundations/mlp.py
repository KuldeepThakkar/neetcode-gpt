import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        a= x
        layers= len(weights)

        for i in range(layers):
            b=np.array(biases[i], dtype="float64")
            w=np.array(weights[i],dtype="float64")

            out = np.dot(a,w)+b

            if i<layers-1:
                a=np.maximum(0.0,out)
            else:
                a=out

        return np.round(a, 5)
        
