# Settings

N_RANDOM    = 100_000
N_GRID      = 10_000
N_CANDIDATES = 4096
N_RESTARTS  = 20
KAPPA       = 1.96
ACQ_FUNC    = 'ucb'
DEFAULT_METHOD = 'bayes'


FORCE_GLOBAL = {'function_1'}

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

KAPPA_PER_FUNCTION = {
    'function_1': 4.0,  
    'function_2': 2.0,  
    'function_3': 1.5,
    'function_4': 1.5,
    'function_5': 1.5,
    'function_6': 2.0,
    'function_7': 2.0,
    'function_8': 1.0,  
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

N_CANDIDATES_PER_FUNCTION = {
    'function_1': 8192,  
}

TIGHT_RADIUS_SCALE = {
    'function_1': 0.15,  
    'function_3': 0.25, 
    'function_4': 0.3,   
    'function_5': 1.0,   
    'function_6': 0.4,   
    'function_7': 0.8,   
    'function_8': 0.3,   
}

GLOBAL_SEARCH_INTERVAL = {
    'function_2': 8,   
    'function_3': 8,   
    'function_4': 20,  
    'function_6': 8,  
}

TIGHT_STREAK_THRESHOLD = {
    
    'function_2': 4,   
    'function_4': 0,  
    'function_5': 0,  
    'function_6': 0,   
    'function_7': 0, 
    'function_8': 0,  
}

ACQ_FUNC_PER_FUNCTION = {
    'function_2': 'ts', 
}

LOG_TRANSFORM_FUNCTIONS = {'function_1'}

FIXED_DIMS = {
    'function_5': {1: 1.0, 2: 1.0, 3: 1.0},  
}

ALTERNATIVE_CENTER = {
    'function_3': [0.567655, 0.714575, 0.440917],  
}
ALTERNATIVE_CENTER_STREAK = {
    'function_3': 15, 
}

LOCAL_OPT_PER_FUNCTION = {
    'function_1': 'de',   
    'function_7': 'de',   
    'function_8': 'de', 
}

BASE_PATH    = 'data'
RUN_LOG      = 'run.txt'
RUN_PREV_LOG = 'run_prev.txt'
PLOTS_DIR    = 'plots'
