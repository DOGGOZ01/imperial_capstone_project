import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from utils import sobol_sample, get_bounds, clip_to_bounds
from config import N_CANDIDATES

def method_surrogate(X: np.ndarray, y: np.ndarray,
                     n_candidates: int = N_CANDIDATES * 4) -> np.ndarray:
    
    lower_bounds, upper_bounds = get_bounds(X)
    dims, num_points = X.shape[1], len(y)

    if num_points >= 20:
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500, max_features='sqrt',
            random_state=42
        )
    
    model.fit(X, y)

    candidates = lower_bounds + sobol_sample(n_candidates, dims) * (upper_bounds - lower_bounds)
    predictions = model.predict(candidates)
    best = np.argmax(predictions)
    return clip_to_bounds(candidates[best], lower_bounds, upper_bounds)