import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from methods.method_surrogate import method_surrogate
from config import PLOTS_DIR

def method_manual(X: np.ndarray, y: np.ndarray, folder_name: str = 'function_?') -> np.ndarray:
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    dims = X.shape[1]

    cols = min(dims, 3)
    rows = (dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if dims > 1 else [axes]

    for d in range(dims):
        ax = axes_flat[d]
        scatter = ax.scatter(X[:, d], y, c=y, cmap='viridis',
                             edgecolors='k', linewidths=0.4, s=60)
        ax.set_xlabel(f'x{d+1}', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'dimension {d+1}', fontsize=10)
        plt.colorbar(scatter, ax=ax)

    for d in range(dims, len(axes_flat)):
        axes_flat[d].set_visible(False)

    fig.suptitle(f'{folder_name}: scatter plots x vs y (n={len(y)})', fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f'{folder_name}_scatter.png')
    plt.savefig(plot_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   [manual] Saved plot to {plot_path}")

    # Auto-pick
    return method_surrogate(X, y)