import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from utils import get_bounds, clip_to_bounds
from config import N_GRID


def method_grid(X: np.ndarray, y: np.ndarray, n_grid: int = N_GRID) -> np.ndarray:
    lower_bounds, upper_bounds = get_bounds(X)
    dims = X.shape[1]

    points_per_dim = max(2, int(round(n_grid ** (1.0 / dims))))
    axes = [np.linspace(lower_bounds[i], upper_bounds[i], points_per_dim) for i in range(dims)]
    grid = np.array(np.meshgrid(*axes, indexing='ij')).reshape(dims, -1).T

    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dims), nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-8,
        normalize_y=True, n_restarts_optimizer=5
    )
    gp.fit(X_scaled, y)

    grid_scaled = x_scaler.transform(grid)
    means, stds = gp.predict(grid_scaled, return_std=True)
    scores = means + 1.96 * stds

    best = np.argmax(scores)
    return clip_to_bounds(grid[best], lower_bounds, upper_bounds)
