# Settings

N_RANDOM = 100_000  # how many random points to try
N_GRID = 10_000     # points for grid search
N_CANDIDATES = 2048 # Sobol points for initial sampling (power of 2)
N_RESTARTS = 10     # restarts for optimizer in Bayes
KAPPA = 1.96        # exploration in UCB: higher = more exploration
ACQ_FUNC = 'ucb'    

DEFAULT_METHOD = 'bayes'  

# Override for specific functions
METHOD_PER_FUNCTION = {
    # 'function_1': 'random',
    # 'function_2': 'grid',
    # 'function_3': 'bayes',
    # 'function_4': 'surrogate',
    # 'function_5': 'manual',
    # 'function_6': 'bayes',
    # 'function_7': 'bayes',
    # 'function_8': 'bayes',
}

BASE_PATH = 'data'
RUN_LOG = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR = 'plots'  

BASE_PATH = 'data'
RUN_LOG = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR = 'plots'  