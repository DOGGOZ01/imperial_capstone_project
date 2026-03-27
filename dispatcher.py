import numpy as np
from config import KAPPA
from methods.method_random import method_random
from methods.method_grid import method_grid
from methods.method_bayes import method_bayes
from methods.method_surrogate import method_surrogate
from methods.method_manual import method_manual
from config import N_GRID, N_RANDOM

def adaptive_bayes_params(dims: int, num_points: int): #UCB or EI
    if num_points < 20 or dims >= 6:
        return 'ucb', KAPPA + (1.0 if dims >= 6 else 0.0)
    return 'ei', KAPPA

def run_method(method: str, X: np.ndarray, y: np.ndarray,
               folder_name: str) -> tuple[np.ndarray, str]:
    dims = X.shape[1]
    num_points = len(y)

    if method == 'random':
        return method_random(X, y), f"random (n={N_RANDOM})"

    elif method == 'grid':
        points_per_dim = int(round(N_GRID ** (1.0 / dims)))
        return method_grid(X, y), f"grid ({points_per_dim}^{dims}≈{points_per_dim**dims})"

    elif method == 'bayes':
        acq_type, kappa_val = adaptive_bayes_params(dims, num_points)
        return method_bayes(X, y, kappa=kappa_val, acq_func=acq_type), f"bayes/{acq_type} κ={kappa_val:.2f}"

    elif method == 'surrogate':
        model_name = 'GradBoost' if num_points >= 20 else 'RandomForest'
        return method_surrogate(X, y), f"surrogate ({model_name})"

    elif method == 'manual':
        point = method_manual(X, y, folder_name)
        return point, f"manual → plots/{folder_name}_scatter.png"

    else:
        raise ValueError(f"Method error")