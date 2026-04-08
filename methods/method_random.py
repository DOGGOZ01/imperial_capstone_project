import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from utils import get_bounds, clip_to_bounds
from config import N_RANDOM

def method_random(X: np.ndarray, y: np.ndarray, n_samples: int = N_RANDOM) -> np.ndarray:
    lower_bounds, upper_bounds = get_bounds(X)
    dims = X.shape[1]

    candidates = np.random.uniform(lower_bounds, upper_bounds, size=(n_samples, dims))

    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dims), nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-8,
        normalize_y=True, n_restarts_optimizer=5
    )
    gp.fit(X_scaled, y)

    cands_scaled = x_scaler.transform(candidates)
    means, stds = gp.predict(cands_scaled, return_std=True)
    scores = means + 1.96 * stds  

    best = np.argmax(scores)
    return clip_to_bounds(candidates[best], lower_bounds, upper_bounds)