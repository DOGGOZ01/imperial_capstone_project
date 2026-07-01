# Black-Box Optimisation - Imperial Capstone Project

Eight unknown functions live in `data/function_1` through `data/function_8` and need to be maximised. The only access to each one is a weekly oracle query: submit a point in `[0,1]^D` and get one scalar number back. No gradient, no analytic form and no internal structure is available, so every decision about where to sample next has to come from a surrogate model fit on whatever has been observed so far.

BBO problems like this show up in hyperparameter tuning, drug discovery and materials design: anywhere an evaluation is expensive enough that you cannot afford to explore blindly. The core tension is always the same: spend the next query exploring an unknown region or refining a region that already looks good.

---

## Documentation

### [Results](doc/results.md)
Current best value and best point per function, plus per-function shape analysis built from real numbers in the accumulated oracle data.

### [Rounds](doc/rounds.md)
Week-by-week changelog across all 12 rounds: what changed in the solver configuration and why, tied directly to the oracle responses that drove each change.

### [Examples](doc/examples.md)
Command output, plot descriptions and the difference between `history.json` (solver state) and `oracle_history.json` (submission record).

### [Datasheet](doc/datasheet.md)
Dataset composition, collection process, preprocessing and known sampling gaps.

### [Model Card](doc/model_card.md)
Solver design summary, intended use and known limitations.

---

## The eight functions

| Function | D | Domain | Current best |
|---|---|---|---|
| function_1 | 2 | Radiation field, narrow peak near zero elsewhere | 6.69e-11 |
| function_2 | 2 | Noisy ML model score, stochastic outputs | 0.7628 |
| function_3 | 3 | Drug discovery scoring, negative by design | -0.00852 |
| function_4 | 4 | Warehouse placement, many local optima | 0.7307 |
| function_5 | 4 | Chemical yield, boundary-seeking optimum | 4457.29 |
| function_6 | 5 | Recipe scoring, negative by design | -0.2214 |
| function_7 | 6 | ML hyperparameter surface | 2.6248 |
| function_8 | 8 | High-dimensional ML performance evaluation | 9.9193 |

Full best points and per-function shape analysis: **[doc/results.md](doc/results.md)**.

Each function ships two NumPy arrays, `initial_inputs.npy` (shape `(N, D)`) and `initial_outputs.npy` (shape `(N,)`). A submission is a hyphen-separated string of floats, one per dimension:

```
0.547332-0.244532                              # function_1 (D=2)
0.014749-0.018820-...-0.848244                 # function_8 (D=8)
```

---

## Running it

```
python main.py             # dispatch every function through its configured method, log to run.txt
python main.py --manual    # generate diagnostic plots for every function instead, no method changes
python dimension_checker.py  # quick N / D / max-y printout per function
python reset_history.py      # wipe solver state (best_x, streak, tight_count) back to defaults
```

---

## Methods

`dispatcher.py` reads `history.json` to decide, per function, whether to search globally over `[0,1]^D` or tightly around the current best point, which acquisition function to use and what kappa, alpha and kernel smoothness to fit the GP with. Every function currently runs through `method_bayes.py`, a Gaussian Process with a Matern kernel, Sobol candidate sampling and UCB, EI or Thompson Sampling refined by L-BFGS-B or Differential Evolution. Two other surrogates exist for comparison: a tree-based one and a PyTorch MC-Dropout MLP, alongside the original random and grid baselines. Current per-function assignments live in [config.py](config.py); the reasoning behind specific parameter choices over time is in [doc/rounds.md](doc/rounds.md).

### Dispatcher logic (`dispatcher.py`)

Before any method runs, the dispatcher answers three questions from `history.json` and `config.py`:

**Tight or global search?** `should_use_tight_search()` returns tight mode once `no_improvement_streak` reaches a function's `TIGHT_STREAK_THRESHOLD` (0 for every function as of round 12, so tight mode activates immediately after any round without improvement), unless the function is in `FORCE_GLOBAL` or `tight_count` has hit `GLOBAL_SEARCH_INTERVAL` and needs a periodic reset.

**Where is the search centred?** `get_search_center()` uses `best_x` from history by default. If a function's `no_improvement_streak` exceeds its `ALTERNATIVE_CENTER_STREAK` (only function_3 has one, set to 15) the centre jumps to a pre-registered fallback point instead, on the theory that the current best region is a local trap.

**Which acquisition function and kappa?** `adaptive_bayes_params()` picks a fixed acquisition from `ACQ_FUNC_PER_FUNCTION` if one is set for the function, otherwise falls back to `ucb` when `num_points < 20` or `dims >= 6` and to `ei` otherwise. Kappa comes straight from `KAPPA_PER_FUNCTION`.

For function_5, the dispatcher also slices out the dimensions listed in `FIXED_DIMS` before calling `method_bayes`, so the GP trains on the free dimensions only, then reassembles the full-dimensional point before returning it.

After a method returns a point, `update_history()` compares the new best observed y against the stored `best_value`. If it is better, `best_x`, `best_value` and `no_improvement_streak=0` are written; otherwise the streak increments. `tight_count` increments on tight rounds and resets to 0 on global rounds, which is what `GLOBAL_SEARCH_INTERVAL` checks against.

### `method_bayes.py` - Gaussian Process Bayesian Optimisation

The main method, used by every function as of round 12.

1. Fits a `GaussianProcessRegressor` with kernel `ConstantKernel * Matern(nu)` on all accumulated `(X, y)` pairs, `alpha` and `nu` taken from `GP_ALPHA_PER_FUNCTION` / `MATERN_NU_PER_FUNCTION`. Inputs are used raw since they already live in `[0,1]^D`, no scaler needed.
2. Draws Sobol candidates: `n_candidates` in global mode (4096 by default, up to 8192 for function_1), or half that (capped at 1024) inside the tight window when `tight_search=True`.
3. In tight mode, the window is a hypercube of radius `0.15/sqrt(D) * tight_radius_scale` around the search centre. For function_7 (D=6) with `tight_radius_scale=0.8` that radius is about 0.049 per dimension; for function_1 (D=2) with scale 0.3 it is about 0.032.
4. Scores every candidate with the chosen acquisition:
   - `ucb = mean + kappa * std`
   - `ei` (expected improvement) weighs the probability and magnitude of beating the current best y
   - `ts` (Thompson Sampling) draws one sample from the GP posterior via `gp.sample_y()` and takes its argmax directly, skipping the refinement step entirely
5. For `ucb`/`ei`, refines the best candidates with either L-BFGS-B multi-start (default, 20 starts from the top-scoring candidates plus the centre) or `scipy.optimize.differential_evolution` when `local_optimizer='de'` (functions 1, 7 and 8, where L-BFGS-B kept converging to the same local optimum).
6. Clips the result to the tight window if applicable, then to `[0,1]^D`.

`verbose=True` (the dispatcher's default) prints the fitted kernel's length scales and log marginal likelihood after every fit, flagging when a length scale sits at its optimisation bound, a sign the Matern smoothness assumption may not fit that function.

### `method_random.py` and `method_grid.py` - Baselines

Both fit a plain GP (`Matern(nu=2.5)`, fixed `alpha=1e-8`) on `MinMaxScaler`-normalised data and score candidates with a fixed UCB (`kappa=1.96`). `method_random` draws `N_RANDOM=100_000` uniform candidates; `method_grid` builds a regular grid with roughly `N_GRID=10_000` total points (`round(N_GRID ** (1/D))` per axis). Neither supports tight search, per-function tuning or acquisition switching. These were the Round 1 baseline and are kept for comparison but are not assigned to any function in the current config.

### `method_surrogate.py` - Tree-Based Surrogate

Replaces the GP with `GradientBoostingRegressor` (300 estimators, depth 4) when `num_points >= 20`, or `RandomForestRegressor` (500 estimators) below that. Scores `N_CANDIDATES * 4` Sobol candidates over the full `[0,1]^D` domain and returns the argmax prediction directly, no acquisition function or uncertainty term. Trees do not need kernel tuning and scale better with dimension than a GP, but with no uncertainty estimate they cannot distinguish "unexplored" from "predicted low". That is why the config moved away from this method for functions 4, 6 and 8 once GP tight search proved better at exploiting a known good region.

### `method_neural.py` - MC Dropout MLP

A PyTorch MLP (64 to 64 to 32 to 1, ReLU, 15% dropout) trained for 1000 epochs with Adam and cosine annealing. At inference, dropout is kept active (`model.train()` instead of `model.eval()`) and 50 stochastic forward passes are run over Sobol candidates; the spread across passes approximates predictive uncertainty the way an ensemble would. Mean and std feed into `ucb = mean + kappa * std`. Falls back to `method_surrogate` if `torch` is not installed. Tried on functions 7 and 8 in round 5, function_7 kept it through round 6 (a genuine +8.9% gain) before also reverting to `method_bayes` in round 7 once tight search made GP-BO competitive again. Not assigned to any function in the current config.

### `method_manual.py` - Diagnostics, Not Optimisation

Triggered by `python main.py --manual`. Produces five image types per function under `plots/`, described with real output examples in [doc/examples.md](doc/examples.md), then falls back to `method_surrogate` for the returned point (the point itself is not meant to be submitted; this mode exists purely to look at the data). It does not read or write tight/global state beyond the same `update_history` bookkeeping every method performs.

---

## Repository layout

| Path | Role |
|---|---|
| `main.py` | Entry point. Loads data per function, merges oracle history, dispatches, writes `run.txt`. |
| `dispatcher.py` | Decides tight vs global search, acquisition function and GP hyperparameters per function; updates `history.json`. |
| `config.py` | Every hyperparameter and per-function override: kappa, GP alpha, Matern nu, tight radius, global reset interval, acquisition, fixed dimensions. |
| `utils.py` | Sobol sampling, point clipping, output string formatting. |
| `logger.py` | Run log with rotation to `run_prev.txt`. |
| `methods/` | `method_bayes.py`, `method_random.py`, `method_grid.py`, `method_surrogate.py`, `method_neural.py`, `method_manual.py`. See [Methods](#methods) above. |
| `data/function_N/` | `initial_inputs.npy`, `initial_outputs.npy` per function. |
| `oracle_history.json` | Every point submitted to the oracle and its response, across all rounds. |
| `history.json` | Per-function solver state: `best_value`, `best_x`, `recommended_x`, `no_improvement_streak`, `tight_count`. |
| `plots/` | Diagnostic images from `--manual` mode. |
| `doc/` | [results.md](doc/results.md), [rounds.md](doc/rounds.md), [examples.md](doc/examples.md), [datasheet.md](doc/datasheet.md), [model_card.md](doc/model_card.md). |
