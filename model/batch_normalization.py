import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        if training:
            mean_batch = np.mean(x, axis=0, keepdims=False)
            var_batch = np.var(x, axis=0, keepdims=False)
            x_hat = (x - mean_batch)/np.sqrt(var_batch + eps)
            running_mean = (1 - momentum)*np.array(running_mean) + momentum*np.array(mean_batch)
            running_var = (1 - momentum)*np.array(running_var) + momentum*np.array(var_batch)
        else:
            x_hat = (np.array(x) - np.array(running_mean)) / np.sqrt(np.array(running_var) + eps)

        y = gamma*x_hat + beta
        return np.round(y, 4), np.round(running_mean, 4), np.round(running_var, 4)