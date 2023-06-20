
from typing import Callable
import numpy as np

def finite_difference(f: Callable[[float], np.ndarray], t_values: np.ndarray, order: int = 1):
    dt = t_values[1] - t_values[0]
    f_values = np.array([f(t) for t in t_values])
    if order == 1:
        return (f_values[2:] - f_values[:-2]) / (2 * dt)
    elif order == 2:
        return (f_values[:-2] - 2 * f_values[1:-1] + f_values[2:]) / (dt**2)
    else:
        raise ValueError("Order must be 1 or 2")

def normalized(vectors: np.ndarray):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
