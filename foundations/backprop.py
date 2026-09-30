import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        z= np.dot(x,w)+b
        y_hat= 1/(1+np.exp(-z))
        loss= 0.5*((y_hat-y_true)**2)
        dz_dy= y_hat-y_true
        dy_dz= y_hat*(1-y_hat)
        delta= dz_dy*dy_dz
        dl_ddw=delta*x
        dl_db=delta*1.0
        return np.round(dl_ddw,5) , float(np.round(dl_db, 5))
       
