# Settings

N_RANDOM    = 100_000
N_GRID      = 10_000
N_CANDIDATES = 4096
N_RESTARTS  = 20
KAPPA       = 1.96
ACQ_FUNC    = 'ucb'
DEFAULT_METHOD = 'bayes'

# function_1: radiation field — very narrow peak, near-zero everywhere else
#             log-transform applied in main.py; force global search always
# function_2: noisy ML model — many local peaks, stochastic outputs
# function_3: drug discovery — negative by design (negative of side effects)
# function_4: warehouse placement — full of local optima
# function_5: chemical yield — unimodal, single global peak
# function_6: cake recipe — negative by design, maximise toward zero
# function_7: ML hyperparameter tuning — 6D, smooth-ish landscape
# function_8: 8D ML model — complex, many local maxima

FORCE_GLOBAL = {'function_1'}

METHOD_PER_FUNCTION = {
    'function_1': 'bayes',
    'function_2': 'bayes',
    'function_3': 'bayes',
    'function_4': 'bayes',
    'function_5': 'bayes',
    'function_6': 'bayes',
    'function_7': 'neural',     # PyTorch MLP surrogate for ML hyperparameter tuning
    'function_8': 'surrogate',  # back to GBDT — neural gave only +0.17% at N=46
}

# UCB exploration coefficient per function
KAPPA_PER_FUNCTION = {
    'function_1': 6.0, 
    'function_2': 2.0,  # raised from 1.5 — stochastic function needs broader exploration
    'function_3': 1.5,
    'function_4': 1.5,  
    'function_5': 1.5,  
    'function_6': 2.0,  
    'function_7': 2.0,  
    'function_8': 2.0,   
}

GP_ALPHA_PER_FUNCTION = {
    'function_1': 1e-4,
    'function_2': 0.05,
    'function_3': 1e-6,
    'function_4': 1e-6,
    'function_5': 1e-6,
    'function_6': 1e-6,
    'function_7': 1e-4,
    'function_8': 1e-4,
}

# Matern smoothness per function
#   nu=2.5 → smooth (unimodal, Gaussian-like peaks)
#   nu=1.5 → rougher (many local optima, noisy)
MATERN_NU_PER_FUNCTION = {
    'function_1': 2.5,   
    'function_2': 1.5,   
    'function_3': 2.5,  
    'function_4': 1.5,  
    'function_5': 2.5,  
    'function_6': 2.5,  
    'function_7': 2.5,   
    'function_8': 1.5,  
}

# Sobol candidate count per function (overrides N_CANDIDATES if set)
#   f1: 2D global search — more candidates for finer coverage of narrow peak
N_CANDIDATES_PER_FUNCTION = {
    'function_1': 8192,
}

# Tight search radius multiplier (base = 0.15 / sqrt(dims))
TIGHT_RADIUS_SCALE = {
    'function_1': 1.0,
    'function_2': 1.5,   
    'function_3': 0.6,   
    'function_4': 0.6,   
    'function_5': 1.0,
    'function_6': 1.0,
    'function_7': 1.0,
    'function_8': 1.0,
}

# Periodically force global search to escape local maxima
#   f2: many local peaks — reset to global every 3 tight searches
#   f3: risk of corner trap — reset every 4 tight searches
GLOBAL_SEARCH_INTERVAL = {
    'function_2': 3,
    'function_3': 4,
}

# Minimum no_improvement_streak before switching to tight search (default 2)
#   f2: stochastic outputs — streak of 2 is too aggressive, need more evidence
TIGHT_STREAK_THRESHOLD = {
    'function_2': 4,
}

# Per-function acquisition function override
#   f2: Thompson Sampling — robust for noisy/multimodal, no kappa to tune
ACQ_FUNC_PER_FUNCTION = {
    'function_2': 'ts',
}

LOG_TRANSFORM_FUNCTIONS = {'function_1'}

BASE_PATH    = 'data'
RUN_LOG      = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR    = 'plots'
