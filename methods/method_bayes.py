import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.preprocessing import MinMaxScaler
from utils import sobol_sample, clip_to_bounds
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


def method_bayes(
    X: np.ndarray,
    y: np.ndarray,
    kappa: float = KAPPA,
    n_candidates: int = N_CANDIDATES,
    n_restarts: int = N_RESTARTS,
    acq_func: str = ACQ_FUNC,
    tight_search: bool = False,
    best_x_known: np.ndarray | None = None,  
) -> np.ndarray:

    dims = X.shape[1]
    lower_bounds = np.zeros(dims)
    upper_bounds = np.ones(dims)

    # Fit scaler on unit hypercube so it acts as identity on [0,1]^D
    x_scaler = MinMaxScaler()
    x_scaler.fit(np.vstack([lower_bounds, upper_bounds]))
    X_scaled = x_scaler.transform(X)

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

    if best_x_known is not None:
        center_original = np.clip(best_x_known, lower_bounds, upper_bounds)
    else:
        center_original = X[current_best_idx]

    center_scaled = x_scaler.transform(center_original.reshape(1, -1)).ravel()

    if tight_search:
        radius = np.mean(upper_bounds - lower_bounds) * 0.15 / np.sqrt(dims)
        adaptive_candidates = min(n_candidates // 2, 1024)

        candidates_local = center_original + np.random.randn(adaptive_candidates, dims) * radius
        candidates = np.clip(candidates_local, lower_bounds, upper_bounds)

        bounds = [
            (max(lower_bounds[d], center_original[d] - radius),
             min(upper_bounds[d], center_original[d] + radius))
            for d in range(dims)
        ]
    else:
        adaptive_candidates = min(n_candidates, max(1024, n_candidates // max(1, dims // 4)))
        candidates_sobol = sobol_sample(adaptive_candidates, dims)
        candidates = lower_bounds + candidates_sobol * (upper_bounds - lower_bounds)

        bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))

    candidates_scaled = x_scaler.transform(candidates)
    means, stds = gp.predict(candidates_scaled, return_std=True)
    acq_values = ucb(means, stds, kappa) if acq_func == 'ucb' else ei(means, stds, current_best_y)

    top_idx = np.argsort(acq_values)[-n_restarts:]
    best_starts_original = candidates[top_idx]
    best_starts_scaled = x_scaler.transform(best_starts_original)

    best_starts_scaled = np.vstack([best_starts_scaled, center_scaled.reshape(1, -1)])

    if tight_search:
        center_norm = center_scaled
        radius_norm = radius / (upper_bounds - lower_bounds).mean()
        bounds_norm = [
            (max(0.0, center_norm[d] - radius_norm),
             min(1.0, center_norm[d] + radius_norm))
            for d in range(dims)
        ]
    else:
        bounds_norm = [(0.0, 1.0)] * dims

    best_acq = -np.inf
    best_x_scaled = best_starts_scaled[-1]  

    for start in best_starts_scaled:
        result = minimize(
            acq_negative, start,
            args=(gp, kappa, current_best_y, acq_func, x_scaler),
            method='L-BFGS-B',
            bounds=bounds_norm,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        val = -result.fun
        if val > best_acq:
            best_acq = val
            best_x_scaled = result.x

    best_original = x_scaler.inverse_transform(best_x_scaled.reshape(1, -1)).ravel()
    return clip_to_bounds(best_original, lower_bounds, upper_bounds)