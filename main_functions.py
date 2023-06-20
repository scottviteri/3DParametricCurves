
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

def create_t_values(n: int = 1000):
    return np.linspace(0, 1, n)

def parametric_representation(c: Callable[[float], np.ndarray], t_values: np.ndarray):
    return np.array([c(t) for t in t_values])

def plot_curve(points: np.ndarray):
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1])
    plt.axis('equal')
    plt.show()

def plot_parametric_curve(c: Callable[[float], np.ndarray], n: int = 1000):
    t_values = create_t_values(n)
    points = parametric_representation(c, t_values)
    plot_curve(points)

def n_gon(n: int):
    def c(t):
        i = int(t * n)
        theta1 = 2 * np.pi * i / n
        theta2 = 2 * np.pi * (i + 1) / n
        alpha = t * n - i
        return (1 - alpha) * np.array([np.cos(theta1), np.sin(theta1)]) + alpha * np.array([np.cos(theta2), np.sin(theta2)])
    return c
