import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.preprocessing import MinMaxScaler
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
    
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dims), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=5)
    gp.fit(X_scaled, y)
    
    grid_scaled = x_scaler.transform(grid)
    means, stds = gp.predict(grid_scaled, return_std=True)
    
    scores = (distances / (distances.max() + 1e-9)) + stds
    best_candidates_idx = np.argsort(scores)[-max(3, dims):]
    
    best = best_candidates_idx[-1] 
    
    return clip_to_bounds(grid[best], lower_bounds, upper_bounds)