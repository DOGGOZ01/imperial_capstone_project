import numpy as np
from utils import get_bounds, clip_to_bounds
from config import N_GRID

def method_grid(X: np.ndarray, y: np.ndarray, n_total: int = N_GRID) -> np.ndarray:
    lower_bounds, upper_bounds = get_bounds(X)
    dims = X.shape[1]
    points_per_dim = max(2, int(round(n_total ** (1.0 / dims))))

    axes = [np.linspace(lower_bounds[d], upper_bounds[d], points_per_dim) for d in range(dims)]
    grid = np.array(np.meshgrid(*axes)).reshape(dims, -1).T

    distances = np.min(
        np.linalg.norm(grid[:, None, :] - X[None, :, :], axis=2),
        axis=1
    )
    best = np.argmax(distances)
    return clip_to_bounds(grid[best], lower_bounds, upper_bounds)