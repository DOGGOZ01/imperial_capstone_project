import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from utils import sobol_sample, clip_to_bounds
from config import KAPPA, N_CANDIDATES, N_RESTARTS, ACQ_FUNC


def ucb(mean, std, kappa):
    return mean + kappa * std


def ei(mean, std, current_best_y):
    z = (mean - current_best_y) / (std + 1e-9)
    return (mean - current_best_y) * norm.cdf(z) + std * norm.pdf(z)


def acq_negative(x_flat, gp, kappa, current_best_y, acq_func):
    mean, std = gp.predict(x_flat.reshape(1, -1), return_std=True)
    val = ucb(mean[0], std[0], kappa) if acq_func == 'ucb' else ei(mean[0], std[0], current_best_y)
    return -val


def _log_kernel(gp: GaussianProcessRegressor, label: str) -> None:
    try:
        ls  = gp.kernel_.k2.length_scale
        lb, ub = gp.kernel_.k2.length_scale_bounds.T
        at_lb = np.any(np.isclose(ls, lb, rtol=1e-2))
        at_ub = np.any(np.isclose(ls, ub, rtol=1e-2))
        warn  = " !! ls AT BOUND" if (at_lb or at_ub) else ""
        scale = gp.kernel_.k1.constant_value
        lml   = gp.log_marginal_likelihood_value_
        ls_str = " ".join(f"{v:.3f}" for v in np.atleast_1d(ls))
        print(f"  [GP:{label}] scale={scale:.3f}  ls=[{ls_str}]  lml={lml:.2f}{warn}")
    except AttributeError:
        pass


def method_bayes(
    X: np.ndarray,
    y: np.ndarray,
    kappa: float = KAPPA,
    n_candidates: int = N_CANDIDATES,
    n_restarts: int = N_RESTARTS,
    acq_func: str = ACQ_FUNC,
    tight_search: bool = False,
    best_x_known: np.ndarray | None = None,
    tight_radius_scale: float = 1.0,
    gp_alpha: float = 1e-8,
    matern_nu: float = 2.5,
    verbose: bool = False,
    label: str = '',
) -> np.ndarray:

    dims = X.shape[1]
    lower_bounds = np.zeros(dims)
    upper_bounds = np.ones(dims)

    length_scale_lower = max(1e-3, 0.05 / np.sqrt(dims))
    length_scale_upper = min(1000.0, 20.0 * np.sqrt(dims))

    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(dims),
        length_scale_bounds=(length_scale_lower, length_scale_upper),
        nu=matern_nu,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=gp_alpha,
        normalize_y=True,
        n_restarts_optimizer=8,
    )
    gp.fit(X, y)  # X is already in [0,1]

    if verbose:
        _log_kernel(gp, label)

    current_best_y = y.max()
    center = (
        np.clip(best_x_known, lower_bounds, upper_bounds)
        if best_x_known is not None
        else X[np.argmax(y)]
    )

    if tight_search:
        radius = 0.15 / np.sqrt(dims) * tight_radius_scale
        adaptive_candidates = min(n_candidates // 2, 1024)

        base = sobol_sample(adaptive_candidates, dims)
        candidates = np.clip(center + (base - 0.5) * 2 * radius, lower_bounds, upper_bounds)

        bounds_opt = [
            (max(0.0, center[d] - radius), min(1.0, center[d] + radius))
            for d in range(dims)
        ]
        tight_lb = np.array([b[0] for b in bounds_opt])
        tight_ub = np.array([b[1] for b in bounds_opt])
    else:
        radius = None
        adaptive_candidates = min(n_candidates, max(1024, n_candidates // max(1, dims // 4)))
        candidates = sobol_sample(adaptive_candidates, dims)
        bounds_opt = [(0.0, 1.0)] * dims
        tight_lb = tight_ub = None

    # Thompson Sampling: draw one posterior sample, return its argmax — no L-BFGS-B needed
    if acq_func == 'ts':
        sample = gp.sample_y(candidates, n_samples=1).ravel()
        best_x = candidates[np.argmax(sample)]
        if tight_search and tight_lb is not None:
            best_x = np.clip(best_x, tight_lb, tight_ub)
        return clip_to_bounds(best_x, lower_bounds, upper_bounds)

    means, stds = gp.predict(candidates, return_std=True)
    acq_values = ucb(means, stds, kappa) if acq_func == 'ucb' else ei(means, stds, current_best_y)

    top_idx = np.argsort(acq_values)[-n_restarts:]
    best_starts = np.vstack([candidates[top_idx], center.reshape(1, -1)])

    best_acq = -np.inf
    best_x = center.copy()

    for start in best_starts:
        result = minimize(
            acq_negative, start,
            args=(gp, kappa, current_best_y, acq_func),
            method='L-BFGS-B',
            bounds=bounds_opt,
            options={'maxiter': 300, 'ftol': 1e-12},
        )
        if -result.fun > best_acq:
            best_acq = -result.fun
            best_x = result.x

    if tight_search and tight_lb is not None:
        best_x = np.clip(best_x, tight_lb, tight_ub)

    return clip_to_bounds(best_x, lower_bounds, upper_bounds)
