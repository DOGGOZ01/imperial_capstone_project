import numpy as np
import os

def check_dimensions():
    base_path = 'data'
    
    print(f"{'Func':<12} | {'Points (N)':<10} | {'Dimensionality (D)':<15} | {'Max Y':<10}")
    print("-" * 55)

    for i in range(1, 9):
        folder_name = f'function_{i}'
        inputs_path = os.path.join(base_path, folder_name, 'initial_inputs.npy')
        outputs_path = os.path.join(base_path, folder_name, 'initial_outputs.npy')

        if os.path.exists(inputs_path) and os.path.exists(outputs_path):
            try:
                X = np.load(inputs_path)
                Y = np.load(outputs_path)
                
                n_points, dims = X.shape
                max_y = np.max(Y)

                print(f"{folder_name:<12} | {n_points:<10} | {dims:<15} | {max_y:.4f}")
            except Exception as e:
                print(f"{folder_name:<12} | Error reading files: {e}")
        else:
            print(f"{folder_name:<12} | Files not found at path: {inputs_path}")

if __name__ == "__main__":
    check_dimensions()