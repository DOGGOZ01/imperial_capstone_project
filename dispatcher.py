import numpy as np
import json
import os
from config import KAPPA
from methods.method_random import method_random
from methods.method_grid import method_grid
from methods.method_bayes import method_bayes
from methods.method_surrogate import method_surrogate
from methods.method_manual import method_manual
from config import N_GRID, N_RANDOM

HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def should_use_tight_search(folder_name: str, current_best_y: float) -> bool:
    history = load_history()
    
    if folder_name not in history:
        return False
    
    prev_best = history[folder_name].get('best_value', 0)
    
    improvement = current_best_y - prev_best
    
    relative_improvement = abs(improvement) / (abs(prev_best) + 1e-10)
    
    no_progress = improvement < 1e-6 or relative_improvement < 0.005
    
    return no_progress

def update_history(folder_name: str, current_best_y: float):
    history = load_history()
    
    if folder_name not in history:
        history[folder_name] = {}
    
    history[folder_name]['best_value'] = float(current_best_y)
    save_history(history)

def adaptive_bayes_params(dims: int, num_points: int):
    if num_points < 20 or dims >= 6:
        return 'ucb', KAPPA + (1.0 if dims >= 6 else 0.0)
    return 'ei', KAPPA

def run_method(method: str, X: np.ndarray, y: np.ndarray,
               folder_name: str) -> tuple[np.ndarray, str]:
    dims = X.shape[1]
    num_points = len(y)
    current_best_y = y.max()

    if method == 'random':
        return method_random(X, y), f"random (n={N_RANDOM})"

    elif method == 'grid':
        points_per_dim = int(round(N_GRID ** (1.0 / dims)))
        return method_grid(X, y), f"grid ({points_per_dim}^{dims}≈{points_per_dim**dims})"

    elif method == 'bayes':
        acq_type, kappa_val = adaptive_bayes_params(dims, num_points)
        tight = should_use_tight_search(folder_name, current_best_y)
        mode = "tight" if tight else "global"
        result = method_bayes(X, y, kappa=kappa_val, acq_func=acq_type, tight_search=tight)
        update_history(folder_name, current_best_y)
        return result, f"bayes/{acq_type}({mode}) κ={kappa_val:.2f}"

    elif method == 'auto':
        if dims <= 3:
            return method_grid(X, y), f"auto/grid"
        else:
            acq_type, kappa_val = adaptive_bayes_params(dims, num_points)
            tight = should_use_tight_search(folder_name, current_best_y)
            mode = "tight" if tight else "global"
            result = method_bayes(X, y, kappa=kappa_val, acq_func=acq_type, tight_search=tight)
            update_history(folder_name, current_best_y)
            return result, f"auto/bayes({mode}) κ={kappa_val:.2f}"

    elif method == 'surrogate':
        model_name = 'GradBoost' if num_points >= 20 else 'RandomForest'
        result = method_surrogate(X, y)
        update_history(folder_name, current_best_y)
        return result, f"surrogate ({model_name})"

    elif method == 'manual':
        point = method_manual(X, y, folder_name)
        update_history(folder_name, current_best_y)
        return point, f"manual → plots/{folder_name}_scatter.png"

    else:
        raise ValueError(f"Method error")