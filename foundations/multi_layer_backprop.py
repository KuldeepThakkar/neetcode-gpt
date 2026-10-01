from typing import List
import numpy as np


class Solution:

    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Convert inputs to NumPy arrays
        x_arr = np.array(x)  # shape: (input_dim,)
        W1_arr = np.array(W1)  # shape: (hidden_dim, input_dim)
        b1_arr = np.array(b1)  # shape: (hidden_dim,)
        W2_arr = np.array(W2)  # shape: (output_dim, hidden_dim)
        b2_arr = np.array(b2)  # shape: (output_dim,)
        y_true_arr = np.array(y_true)  # shape: (output_dim,)

        N = len(y_true)

        # --- Forward Pass (W @ x + b) ---
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0.0, z1)
        z2 = np.dot(W2_arr, a1) + b2_arr
        loss = np.mean((z2 - y_true_arr) ** 2)

        # --- Backward Pass ---
        # 1. Output Layer Gradients
        dz2 = (2.0 / N) * (z2 - y_true_arr)
        # Gradient w.r.t W2 is outer product of dz2 (output_dim,) and a1 (hidden_dim,) -> (output_dim, hidden_dim)
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        # 2. Hidden Layer Gradients
        da1 = np.dot(W2_arr.T, dz2)
        dz1 = da1 * (z1 > 0)  # Backprop through ReLU
        # Gradient w.r.t W1 is outer product of dz1 (hidden_dim,) and x (input_dim,) -> (hidden_dim, input_dim)
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1

        # Return results converted back to lists and rounded to 4 decimals
        return {
            "loss": float(np.round(loss, 4)),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }
