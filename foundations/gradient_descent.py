class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x,iteration,lr= init,iterations,learning_rate
        
        for _ in range(iteration):
            der= 2*x
            x=x-lr*der

        return round(x, 5)
