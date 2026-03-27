import numpy as np
from utils import get_bounds, clip_to_bounds
from config import N_RANDOM

def method_random(X: np.ndarray, y: np.ndarray, n_samples: int = N_RANDOM) -> np.ndarray:

    lower_bounds, upper_bounds = get_bounds(X)
    candidates = np.random.uniform(lower_bounds, upper_bounds, size=(n_samples, X.shape[1]))

    distances = np.min(
        np.linalg.norm(candidates[:, None, :] - X[None, :, :], axis=2),
        axis=1
    )
    best = np.argmax(distances)
    return clip_to_bounds(candidates[best], lower_bounds, upper_bounds)