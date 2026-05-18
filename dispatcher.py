import numpy as np
import json
import os
from config import (
    KAPPA, N_GRID, N_RANDOM,
    KAPPA_PER_FUNCTION, GP_ALPHA_PER_FUNCTION, MATERN_NU_PER_FUNCTION,
    N_CANDIDATES_PER_FUNCTION, TIGHT_RADIUS_SCALE, GLOBAL_SEARCH_INTERVAL,
    FORCE_GLOBAL, TIGHT_STREAK_THRESHOLD, ACQ_FUNC_PER_FUNCTION,
)
from methods.method_random    import method_random
from methods.method_grid      import method_grid
from methods.method_bayes     import method_bayes
from methods.method_surrogate import method_surrogate
from methods.method_neural    import method_neural
from methods.method_manual    import method_manual

HISTORY_FILE = 'history.json'


def load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_history(history: dict):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_history_entry(folder_name: str) -> dict:
    return load_history().get(folder_name, {})


def update_history(folder_name: str, recommended_x: np.ndarray, data_best_y: float,
                   X: np.ndarray, y: np.ndarray, used_tight: bool = False):
    history = load_history()
    if folder_name not in history:
        history[folder_name] = {}

    prev_best = history[folder_name].get('best_value', -np.inf)

    if data_best_y > prev_best:
        history[folder_name]['best_value'] = float(data_best_y)
        history[folder_name]['best_x']     = X[np.argmax(y)].tolist()
        history[folder_name]['was_improved'] = True
        history[folder_name]['no_improvement_streak'] = 0
    else:
        history[folder_name]['was_improved'] = False
        history[folder_name]['no_improvement_streak'] = (
            history[folder_name].get('no_improvement_streak', 0) + 1
        )

    # Track consecutive tight searches so we can reset to global periodically
    if used_tight:
        history[folder_name]['tight_count'] = history[folder_name].get('tight_count', 0) + 1
    else:
        history[folder_name]['tight_count'] = 0

    history[folder_name]['recommended_x'] = recommended_x.tolist()
    save_history(history)


def should_use_tight_search(folder_name: str) -> bool:
    if folder_name in FORCE_GLOBAL:
        return False

    entry = get_history_entry(folder_name)
    if entry.get('best_value', -np.inf) == -np.inf:
        return False

    interval = GLOBAL_SEARCH_INTERVAL.get(folder_name)
    if interval is not None and entry.get('tight_count', 0) >= interval:
        return False

    threshold = TIGHT_STREAK_THRESHOLD.get(folder_name, 2)
    return entry.get('no_improvement_streak', 0) >= threshold


def get_search_center(folder_name: str, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    entry = get_history_entry(folder_name)
    if 'best_x' in entry and entry['best_x'] is not None:
        return np.array(entry['best_x'])
    return X[np.argmax(y)]


def adaptive_bayes_params(folder_name: str, dims: int, num_points: int) -> tuple[str, float]:
    kappa = KAPPA_PER_FUNCTION.get(folder_name, KAPPA)
    if folder_name in ACQ_FUNC_PER_FUNCTION:
        acq = ACQ_FUNC_PER_FUNCTION[folder_name]
    else:
        acq = 'ucb' if (num_points < 20 or dims >= 6) else 'ei'
    return acq, kappa


def run_method(method: str, X: np.ndarray, y: np.ndarray,
               folder_name: str) -> tuple[np.ndarray, str]:
    dims        = X.shape[1]
    num_points  = len(y)
    data_best_y = float(y.max())

    if method == 'random':
        result = method_random(X, y)
        update_history(folder_name, result, data_best_y, X, y)
        return result, f"random (n={N_RANDOM})"

    elif method == 'grid':
        points_per_dim = int(round(N_GRID ** (1.0 / dims)))
        result = method_grid(X, y)
        update_history(folder_name, result, data_best_y, X, y)
        return result, f"grid ({points_per_dim}^{dims}~{points_per_dim**dims})"

    elif method == 'bayes':
        acq_type, kappa_val = adaptive_bayes_params(folder_name, dims, num_points)
        tight        = should_use_tight_search(folder_name)
        center       = get_search_center(folder_name, X, y) if tight else None
        radius_scale = TIGHT_RADIUS_SCALE.get(folder_name, 1.0)
        gp_alpha     = GP_ALPHA_PER_FUNCTION.get(folder_name, 1e-8)
        matern_nu    = MATERN_NU_PER_FUNCTION.get(folder_name, 2.5)
        n_cands      = N_CANDIDATES_PER_FUNCTION.get(folder_name, None)

        kwargs = dict(
            kappa=kappa_val, acq_func=acq_type,
            tight_search=tight, best_x_known=center,
            tight_radius_scale=radius_scale,
            gp_alpha=gp_alpha, matern_nu=matern_nu,
            verbose=True, label=folder_name,
        )
        if n_cands:
            kwargs['n_candidates'] = n_cands

        result = method_bayes(X, y, **kwargs)
        update_history(folder_name, result, data_best_y, X, y, used_tight=tight)
        mode = 'tight' if tight else 'global'
        return result, f"bayes/{acq_type}({mode}) k={kappa_val:.2f} a={gp_alpha:.0e} nu={matern_nu}"

    elif method == 'neural':
        kappa_val = KAPPA_PER_FUNCTION.get(folder_name, KAPPA)
        n_cands   = N_CANDIDATES_PER_FUNCTION.get(folder_name, None)
        kwargs    = dict(kappa=kappa_val)
        if n_cands:
            kwargs['n_candidates'] = n_cands
        result = method_neural(X, y, **kwargs)
        update_history(folder_name, result, data_best_y, X, y)
        return result, f"neural/mc-dropout k={kappa_val:.2f}"

    elif method == 'auto':
        if dims <= 3:
            result = method_grid(X, y)
            update_history(folder_name, result, data_best_y, X, y)
            return result, "auto/grid"
        acq_type, kappa_val = adaptive_bayes_params(folder_name, dims, num_points)
        tight        = should_use_tight_search(folder_name)
        center       = get_search_center(folder_name, X, y) if tight else None
        radius_scale = TIGHT_RADIUS_SCALE.get(folder_name, 1.0)
        result = method_bayes(
            X, y, kappa=kappa_val, acq_func=acq_type,
            tight_search=tight, best_x_known=center,
            tight_radius_scale=radius_scale,
        )
        update_history(folder_name, result, data_best_y, X, y, used_tight=tight)
        return result, f"auto/bayes({'tight' if tight else 'global'}) k={kappa_val:.2f}"

    elif method == 'surrogate':
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        model_name = 'GradBoost' if num_points >= 20 else 'RandomForest'
        result = method_surrogate(X, y)
        update_history(folder_name, result, data_best_y, X, y)
        return result, f"surrogate ({model_name})"

    elif method == 'manual':
        point = method_manual(X, y, folder_name)
        update_history(folder_name, point, data_best_y, X, y)
        return point, f"manual -> plots/{folder_name}_scatter.png"

    else:
        raise ValueError(f"Unknown method: '{method}'")
