import os
import numpy as np
from datetime import datetime

from config import BASE_PATH, RUN_LOG, RUN_PREV_LOG, PLOTS_DIR, METHOD_PER_FUNCTION, DEFAULT_METHOD
from logger import rotate_logs, RunLogger
from utils import format_output
from dispatcher import run_method

rotate_logs()
logger = RunLogger(RUN_LOG)

COL_W = 40
header = f"{'Function':<12} | {'D':<3} | {'N':<5} | {'Method & params':<{COL_W}} | Recommended point"
separator = "─" * (12 + 3 + 5 + COL_W + 60)
logger.log(header)
logger.log(separator)

for i in range(1, 9):
    folder = f'function_{i}'
    x_path = os.path.join(BASE_PATH, folder, 'initial_inputs.npy')
    y_path = os.path.join(BASE_PATH, folder, 'initial_outputs.npy')

    if not os.path.exists(x_path):
        logger.log(f"{folder:<12} | --- | ----- | {'files not found':<{COL_W}} | —")
        continue

    X_data = np.load(x_path)
    y_data = np.load(y_path)
    dims = X_data.shape[1]
    num_points = len(y_data)
    method = METHOD_PER_FUNCTION.get(folder, DEFAULT_METHOD)

    try:
        next_point, desc = run_method(method, X_data, y_data, folder)
        logger.log(f"{folder:<12} | {dims:<3} | {num_points:<5} | {desc:<{COL_W}} | {format_output(next_point)}")
    except Exception as e:
        logger.log(f"{folder:<12} | {dims:<3} | {num_points:<5} | {'ERROR':<{COL_W}} | {e}")

logger.log("")
logger.log(f"# Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.close()

print(f"\n→ Results: '{RUN_LOG}'")
if os.path.exists(RUN_PREV_LOG):
    print(f"→ Previous run: '{RUN_PREV_LOG}'")
if os.path.exists(PLOTS_DIR) and os.listdir(PLOTS_DIR):
    print(f"→ Scatter plots: '{PLOTS_DIR}/'")