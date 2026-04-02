import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.preprocessing import MinMaxScaler
from utils import sobol_sample, get_bounds, clip_to_bounds
from config import KAPPA, N_CANDIDATES, N_RESTARTS, ACQ_FUNC

def ucb(mean, std, kappa):
    return mean + kappa * std

def ei(mean, std, current_best_y):
    z = (mean - current_best_y) / (std + 1e-9)
    return (mean - current_best_y) * norm.cdf(z) + std * norm.pdf(z)

def acq_negative(x_flat, gp, kappa, current_best_y, acq_func, x_scaler):
    x_norm = x_scaler.transform(x_flat.reshape(1, -1))
    mean, std = gp.predict(x_norm, return_std=True)
    val = ucb(mean[0], std[0], kappa) if acq_func == 'ucb' else ei(mean[0], std[0], current_best_y)
    return -val

def method_bayes(X: np.ndarray, y: np.ndarray, kappa: float = KAPPA, n_candidates: int = N_CANDIDATES, n_restarts: int = N_RESTARTS, acq_func: str = ACQ_FUNC, tight_search: bool = False) -> np.ndarray:
    lower_bounds, upper_bounds = get_bounds(X)
    dims = X.shape[1]

    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    length_scale_lower = max(1e-3, 0.05 / np.sqrt(dims))
    length_scale_upper = min(1000.0, 20.0 * np.sqrt(dims))
    
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dims),
        length_scale_bounds=(length_scale_lower, length_scale_upper),
        nu=2.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-8,
        normalize_y=True, n_restarts_optimizer=15
    )
    gp.fit(X_scaled, y)
    current_best_y = y.max()
    current_best_idx = np.argmax(y)
    best_point_normalized = X_scaled[current_best_idx]
    best_point_original = X[current_best_idx]

    if tight_search:
        radius = np.mean(X.max(axis=0) - X.min(axis=0)) * 0.15 / np.sqrt(dims)
        adaptive_candidates = min(n_candidates // 2, 1024)
        
        candidates_local = best_point_original + np.random.randn(adaptive_candidates, dims) * radius
        candidates = np.clip(candidates_local, lower_bounds, upper_bounds)
        
        bounds = [(max(0, best_point_original[d] - radius), min(1, best_point_original[d] + radius)) 
                  for d in range(dims)]
    else:
        adaptive_candidates = min(n_candidates, max(1024, n_candidates // max(1, dims // 4)))
        candidates = sobol_sample(adaptive_candidates, dims)
        bounds = [(0.0, 1.0)] * dims
    
    means, stds = gp.predict(x_scaler.transform(candidates), return_std=True)
    acq_values = ucb(means, stds, kappa) if acq_func == 'ucb' else ei(means, stds, current_best_y)
    best_starts = candidates[np.argsort(acq_values)[-n_restarts:]]

    best_starts = np.vstack([best_starts, best_point_normalized.reshape(1, -1)])

    best_acq = -np.inf
    best_x = best_starts[-1]
    
    for start in best_starts:
        result = minimize(acq_negative, start,
                          args=(gp, kappa, current_best_y, acq_func, x_scaler),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12}) 
        if result.success and -result.fun > best_acq:
            best_acq = -result.fun
            best_x = result.x

    best_original = x_scaler.inverse_transform(best_x.reshape(1, -1)).ravel()
    return clip_to_bounds(best_original, lower_bounds, upper_bounds)