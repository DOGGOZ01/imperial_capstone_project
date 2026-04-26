# Settings

N_RANDOM = 100_000  # how many random points to try
N_GRID = 10_000     # points for grid search
N_CANDIDATES = 4096 # Sobol points for initial sampling (power of 2)
N_RESTARTS = 20     # restarts for optimizer in Bayes
KAPPA = 1.96        # default exploration coefficient for UCB
ACQ_FUNC = 'ucb'
FORCE_GLOBAL = {'function_1'}  # always use global search for function_1
DEFAULT_METHOD = 'bayes'

# Per-function kappa tuning based on observed trends:
#   function_1: all outputs ~0, need extreme exploration to find the peak
#   function_2: moderate, found 0.611, improving with higher x2
#   function_3: best found at corner (0.97, 0.94, 0.45), exploit tightly
#   function_4: just found positive region at center, exploit + small explore
#   function_5: clear trend (x2,x3→1, x1→0), exploit boundary
#   function_6: surrogate was making it worse; switched to bayes, explore carefully
#   function_7: best at (0.06, 0.16, 0.59, 0.25, 0.37, 0.82), exploit tightly
#   function_8: best at week-1 pattern (x1-x4≈0, x5≈1, x6≈0.55), exploit
KAPPA_PER_FUNCTION = {
    'function_1': 6.0,
    'function_2': 2.0,
    'function_3': 1.5,
    'function_4': 2.0,
    'function_5': 1.5,
    'function_6': 2.5,
    'function_7': 1.5,
    'function_8': 1.5,
}

# function_4, 6, 8 switched from surrogate to bayes to enable tight search
METHOD_PER_FUNCTION = {
    'function_1': 'bayes',
    'function_2': 'bayes',
    'function_3': 'bayes',
    'function_4': 'bayes',
    'function_5': 'bayes',
    'function_6': 'bayes',
    'function_7': 'bayes',
    'function_8': 'bayes',
}

BASE_PATH = 'data'
RUN_LOG = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR = 'plots'  

BASE_PATH = 'data'
RUN_LOG = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR = 'plots'  