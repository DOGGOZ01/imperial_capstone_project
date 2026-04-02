import json
import os

HISTORY_FILE = 'history.json'

def reset_history():
    history = {
        f'function_{i}': {'best_value': 0.0, 'was_improved': True}
        for i in range(1, 9)
    }
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History reset")

if __name__ == '__main__':
    reset_history()
