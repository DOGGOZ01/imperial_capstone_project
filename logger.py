import os
import shutil
from datetime import datetime
from config import RUN_LOG, RUN_PREV_LOG, DEFAULT_METHOD, KAPPA, N_CANDIDATES, ACQ_FUNC

def rotate_logs():
    if os.path.exists(RUN_LOG):
        shutil.copy2(RUN_LOG, RUN_PREV_LOG)

class RunLogger:

    def __init__(self, path: str):
        self.file = open(path, 'w', encoding='utf-8')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log(f"# Run: {timestamp}")
        self.log(f"# DEFAULT_METHOD={DEFAULT_METHOD}  KAPPA={KAPPA}"
                 f"  N_CANDIDATES={N_CANDIDATES}  ACQ={ACQ_FUNC}")
        self.log("")

    def log(self, line: str = ''):
        print(line)
        self.file.write(line + '\n')

    def close(self):
        self.file.close()