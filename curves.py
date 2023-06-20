
from typing import Callable
import numpy as np
from main import create_t_values, parametric_representation
from helpers import finite_difference, normalized
from scipy.ndimage import gaussian_filter1d

def smoothed_n_gon(n: int, sigma: float):
    def c_smooth(t):
        points = parametric_representation(n_gon(n), create_t_values())
        smoothed_points = gaussian_filter1d(points, sigma, mode='wrap', axis=0)
        i = int(t * len(smoothed_points))
        if i == len(smoothed_points):  # handle the case when t = 1
            i = 0
        return smoothed_points[i]
    return c_smooth

def circle(t):
    return np.array([3 * np.cos(2*np.pi*t), 3 * np.sin(2*np.pi*t), 0.0])

def deltoid(t):
    theta = 3 * np.pi * t
    return np.array([2 * np.cos(theta) + np.cos(2*theta), 2 * np.sin(theta) - np.sin(2*theta)])
