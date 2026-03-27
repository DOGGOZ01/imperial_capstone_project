import numpy as np
from scipy.stats.qmc import Sobol

def format_output(point: np.ndarray) -> str:
    return "-".join([f"{x:.6f}" for x in point])

def sobol_sample(n: int, dim: int) -> np.ndarray: # Quasi-random Sobol sequence
    return Sobol(d=dim, scramble=True).random(n=n)

def get_bounds(X: np.ndarray):
    return X.min(axis=0), X.max(axis=0)

def clip_to_bounds(point: np.ndarray, lower_bounds, upper_bounds) -> np.ndarray:
    return np.clip(point, lower_bounds, upper_bounds)