import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from methods.method_surrogate import method_surrogate
from config import PLOTS_DIR


def _save_scatter(X, y, folder_name, dims):
    cols = min(dims, 3)
    rows = (dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = np.array(axes).flatten() if dims > 1 else [axes]

    for d in range(dims):
        ax = axes_flat[d]
        scatter = ax.scatter(X[:, d], y, c=y, cmap='viridis',
                             edgecolors='k', linewidths=0.4, s=60)
        best_idx = np.argmax(y)
        ax.scatter(X[best_idx, d], y[best_idx], c='red', marker='*',
                   s=180, zorder=5, label='best')
        ax.set_xlabel(f'x{d+1}', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'dimension {d+1}', fontsize=10)
        ax.legend(fontsize=8)
        plt.colorbar(scatter, ax=ax)

    for d in range(dims, len(axes_flat)):
        axes_flat[d].set_visible(False)

    fig.suptitle(f'{folder_name}: x vs y per dimension (n={len(y)})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{folder_name}_scatter.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   [manual] Scatter    -> {path}")


def _save_table(X, y, folder_name, dims):
    order = np.argsort(y)[::-1]
    col_labels = ['Rank', 'y'] + [f'x{d+1}' for d in range(dims)]

    cell_data = []
    for rank, i in enumerate(order):
        row = [str(rank + 1), f'{y[i]:.6g}'] + [f'{X[i, d]:.4f}' for d in range(dims)]
        cell_data.append(row)

    n_rows = len(cell_data)
    n_cols = len(col_labels)

    norm = mcolors.Normalize(vmin=0, vmax=max(n_rows - 1, 1))
    cmap_table = plt.cm.RdYlGn

    row_colors = []
    for i in range(n_rows):
        color = cmap_table(norm(n_rows - 1 - i))
        row_colors.append([(*color[:3], 0.55)] * n_cols)

    col_width = max(1.1, 6.0 / n_cols)
    fig_width = max(8, n_cols * col_width)
    fig_height = max(3, (n_rows + 1) * 0.38 + 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=row_colors,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)

    header_color = '#2E4057'
    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')

    ax.set_title(
        f'{folder_name}: all {n_rows} points, sorted best → worst',
        fontsize=12, fontweight='bold', pad=8,
    )
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{folder_name}_table.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   [manual] Table      -> {path}")


def _save_parallel_coords(X, y, folder_name, dims):
    n = len(y)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-12)

    cmap = plt.cm.viridis
    x_positions = list(range(dims))

    fig, ax = plt.subplots(figsize=(max(9, dims * 1.6), 5))

    order = np.argsort(y_norm)
    for i in order:
        color = cmap(y_norm[i])
        alpha = 0.25 + 0.65 * y_norm[i]
        lw = 0.5 + 1.8 * y_norm[i]
        ax.plot(x_positions, X[i, :], color=color, alpha=alpha, lw=lw)

    best_idx = np.argmax(y)
    ax.plot(x_positions, X[best_idx, :], color='red', lw=2.5, zorder=5,
            label=f'Best  y={y[best_idx]:.4g}')

    top5 = np.argsort(y)[-5:][::-1]
    for rank, i in enumerate(top5[1:], start=2):
        ax.plot(x_positions, X[i, :], color='orange', lw=1.2, alpha=0.8,
                zorder=4, label=f'Top {rank}  y={y[i]:.4g}' if rank <= 3 else None)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'x{d+1}' for d in range(dims)], fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Value in [0, 1]', fontsize=10)
    ax.set_title(f'{folder_name}: parallel coordinates — color = y rank  (n={n})', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    ax.grid(axis='x', alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y_min, vmax=y_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='y value', shrink=0.8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{folder_name}_parallel.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   [manual] Parallel   -> {path}")


def _save_heatmap_2d(X, y, folder_name):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=0.1, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4,
                                  normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X, y)

    res = 80
    x1g = np.linspace(0, 1, res)
    x2g = np.linspace(0, 1, res)
    xx1, xx2 = np.meshgrid(x1g, x2g)
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    mean, std = gp.predict(grid, return_std=True)
    mean_2d = mean.reshape(res, res)
    std_2d = std.reshape(res, res)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    best_idx = np.argmax(y)

    im1 = axes[0].imshow(mean_2d, origin='lower', extent=[0, 1, 0, 1],
                         cmap='viridis', aspect='auto')
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                    edgecolors='k', linewidths=0.8, s=70, zorder=5)
    axes[0].scatter(X[best_idx, 0], X[best_idx, 1], c='red', marker='*',
                    s=250, zorder=6, label=f'Best ({X[best_idx,0]:.3f},{X[best_idx,1]:.3f})')
    axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')
    axes[0].set_title('GP Mean — predicted landscape')
    axes[0].legend(fontsize=8)
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(std_2d, origin='lower', extent=[0, 1, 0, 1],
                         cmap='plasma', aspect='auto')
    axes[1].scatter(X[:, 0], X[:, 1], c='white', edgecolors='k',
                    linewidths=0.8, s=55, zorder=5)
    axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
    axes[1].set_title('GP Std — unexplored regions (bright = uncertain)')
    plt.colorbar(im2, ax=axes[1])

    fig.suptitle(f'{folder_name}: GP landscape (n={len(y)})', fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{folder_name}_heatmap.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   [manual] GP heatmap -> {path}")


def _save_pairplot(X, y, folder_name, dims):
    """2D projection scatter matrix for dims >= 3."""
    fig, axes = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    best_idx = np.argmax(y)

    for i in range(dims):
        for j in range(dims):
            ax = axes[i][j]
            if i == j:
                ax.hist(X[:, i], bins=8, color='steelblue', alpha=0.7)
                ax.set_xlabel(f'x{i+1}')
                ax.axvline(X[best_idx, i], color='red', lw=1.5)
            else:
                sc = ax.scatter(X[:, j], X[:, i], c=y_norm, cmap='viridis',
                                s=30, alpha=0.8, edgecolors='k', linewidths=0.2)
                ax.scatter(X[best_idx, j], X[best_idx, i], c='red',
                           marker='*', s=150, zorder=5)
            if j == 0:
                ax.set_ylabel(f'x{i+1}', fontsize=8)
            if i == dims - 1:
                ax.set_xlabel(f'x{j+1}', fontsize=8)

    fig.suptitle(f'{folder_name}: pairplot (color=y, red star=best)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f'{folder_name}_pairplot.png')
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"   [manual] Pairplot   -> {path}")


def method_manual(X: np.ndarray, y: np.ndarray, folder_name: str = 'function_?') -> np.ndarray:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    dims = X.shape[1]

    _save_scatter(X, y, folder_name, dims)
    _save_table(X, y, folder_name, dims)
    _save_parallel_coords(X, y, folder_name, dims)

    if dims == 2:
        _save_heatmap_2d(X, y, folder_name)
    else:
        _save_pairplot(X, y, folder_name, dims)

    return method_surrogate(X, y)
