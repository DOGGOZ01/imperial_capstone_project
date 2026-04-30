import numpy as np
from utils import sobol_sample
from config import KAPPA, N_CANDIDATES

try:
    import torch
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    _TORCH_OK = True

except ImportError:
    _TORCH_OK = False


def method_neural(
    X: np.ndarray,
    y: np.ndarray,
    kappa: float = KAPPA,
    n_candidates: int = N_CANDIDATES * 2,
    n_mc: int = 50,
    n_epochs: int = 1000,
) -> np.ndarray:
    if not _TORCH_OK:
        from methods.method_surrogate import method_surrogate
        return method_surrogate(X, y)

    dims = X.shape[1]

    y_mean = float(y.mean())
    y_std  = float(y.std()) + 1e-8
    y_norm = (y - y_mean) / y_std

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y_norm)

    model = _MLP(dims, hidden=64, dropout=0.15)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        nn.MSELoss()(model(X_t), y_t).backward()
        optimizer.step()
        scheduler.step()

    # MC Dropout: keep dropout active for uncertainty estimates
    model.train()
    candidates = torch.FloatTensor(sobol_sample(n_candidates, dims))

    with torch.no_grad():
        preds = torch.stack([model(candidates) for _ in range(n_mc)], dim=0)

    mean = preds.mean(0).numpy() * y_std + y_mean
    std  = preds.std(0).numpy()  * y_std

    ucb_values = mean + kappa * std
    return candidates[int(np.argmax(ucb_values))].numpy()
