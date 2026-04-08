# Settings

N_RANDOM = 100_000  # how many random points to try
N_GRID = 10_000     # points for grid search
N_CANDIDATES = 4096 # Sobol points for initial sampling (power of 2)
N_RESTARTS = 20     # restarts for optimizer in Bayes 
KAPPA = 1.96        # exploration in UCB: higher = more exploration
ACQ_FUNC = 'ucb'    
FORCE_GLOBAL = {'function_1'}  
DEFAULT_METHOD = 'bayes'  

# Override for specific functions
METHOD_PER_FUNCTION = {
    'function_1': 'bayes',     
    'function_2': 'bayes',    
    'function_3': 'bayes',     
    'function_4': 'surrogate', 
    'function_5': 'bayes',     
    'function_6': 'surrogate', 
    'function_7': 'bayes',    
    'function_8': 'surrogate',
}

BASE_PATH = 'data'
RUN_LOG = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR = 'plots'  

BASE_PATH = 'data'
RUN_LOG = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR = 'plots'  