# Settings

N_RANDOM    = 100_000
N_GRID      = 10_000
N_CANDIDATES = 4096
N_RESTARTS  = 20
KAPPA       = 1.96
ACQ_FUNC    = 'ucb'
DEFAULT_METHOD = 'bayes'

# function_1: radiation field — very narrow peak, near-zero everywhere else
#             log-transform applied in main.py; tight search around [0.728, 0.734]
# function_2: noisy ML model — many local peaks, stochastic outputs
# function_3: drug discovery — negative by design (negative of side effects)
# function_4: warehouse placement — sharp peak near [0.4,0.4,0.4,0.4], always tight
# function_5: chemical yield — unimodal, x2=x3=x4=1 confirmed optimal direction
# function_6: cake recipe — negative by design, maximise toward zero
# function_7: ML hyperparameter tuning — 6D, GP-BO tight around [0.071,0.198,0.544,...]
# function_8: 8D ML model — tight around x4≈0, x5≈1, x6≈0.46

# f1 removed from FORCE_GLOBAL — 7 rounds of global found nothing; tight around [0.728,0.734]
FORCE_GLOBAL = set()

METHOD_PER_FUNCTION = {
    'function_1': 'bayes',
    'function_2': 'bayes',
    'function_3': 'bayes',
    'function_4': 'bayes',
    'function_5': 'bayes',
    'function_6': 'bayes',
    'function_7': 'bayes',    # reverted from neural — MLP gave -28% regression at N=43
    'function_8': 'bayes',    # reverted from surrogate — GBDT ignores known best pattern
}

# UCB exploration coefficient per function
KAPPA_PER_FUNCTION = {
    'function_1': 6.0,
    'function_2': 2.0,  # stochastic — needs broader exploration
    'function_3': 1.5,
    'function_4': 1.5,
    'function_5': 1.5,
    'function_6': 2.0,
    'function_7': 2.0,
    'function_8': 2.5,  # raised — explore tight window more thoroughly
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
N_CANDIDATES_PER_FUNCTION = {
    'function_1': 8192,  # 2D — more candidates for narrow peak coverage
}

# Tight search radius multiplier (base = 0.15 / sqrt(dims))
TIGHT_RADIUS_SCALE = {
    'function_1': 1.0,
    'function_2': 1.5,
    'function_3': 0.25,  # reduced from 0.6 — EI was pushing to x1=1.0 boundary
    'function_4': 0.6,
    'function_5': 1.0,
    'function_6': 0.4,   # reduced from 1.0 — tight search was escaping known best
    'function_7': 1.0,
    'function_8': 1.0,
}

# Periodically force global search to escape local maxima
GLOBAL_SEARCH_INTERVAL = {
    'function_2': 3,
    'function_3': 4,
    'function_4': 10,  # always tight, but escape every 10 rounds as safety valve
    'function_6': 8,   # tight_count=7 already, needs a reset path
}

# Minimum no_improvement_streak before switching to tight search (default 2)
TIGHT_STREAK_THRESHOLD = {
    'function_2': 4,   # stochastic — need more evidence before tightening
    'function_4': 0,   # always tight — sharp peak, global rounds waste evaluations
    'function_5': 0,   # always tight — x2=x3=x4=1 confirmed, refine x1 only
    'function_7': 0,   # immediately tight around known best after neural regression
}

# Per-function acquisition function override
ACQ_FUNC_PER_FUNCTION = {
    'function_2': 'ts',  # Thompson Sampling — robust for noisy/multimodal
}

LOG_TRANSFORM_FUNCTIONS = {'function_1'}

BASE_PATH    = 'data'
RUN_LOG      = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR    = 'plots'
