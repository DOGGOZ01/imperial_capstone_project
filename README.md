# Black-Box Optimisation - Imperial Capstone Project

## Project Overview

This project focuses on Black-Box Optimisation (BBO): eight unknown functions need to be maximised using only the input-output pairs returned by an external oracle.

BBO is relevant to a wide range of real-world problems, including hyperparameter tuning, drug discovery, and materials design. Any scenario where evaluations are expensive and the objective cannot be inspected directly falls into this category. The core challenge across all of these is the same: deciding when to explore new regions of the search space versus refining what already looks promising.

---

## Inputs and Outputs

Each function is defined over a D-dimensional unit hypercube and returns a scalar response. The starting dataset varies by function:

| Function | Dimensions (D) | 
|---|---|
| function_1 | 2 | 
| function_2 | 2 | 
| function_3 | 3 | 
| function_4 | 4 | 
| function_5 | 4 | 
| function_6 | 5 | 
| function_7 | 6 | 
| function_8 | 8 | 

Each function ships with two NumPy arrays:
- `initial_inputs.npy` - shape `(N, D)`, values in `[0, 1]^D`
- `initial_outputs.npy` - shape `(N,)`, scalar oracle responses

Each round, one candidate point is submitted per function. The oracle evaluates it, returns a score, and that observation is added to the dataset for the next round. The submitted point is formatted as hyphen-separated floats:

```
0.547332-0.244532          # function_1 (D=2)
0.014749-0.018820-...-0.848244   # function_8 (D=8)
```

---

## Technical Approach

The solver in `dispatcher.py` reads per-function method assignments from `config.py`, calls the appropriate module, and updates `history.json` with the latest best value and submitted point. Each method is described below.

### Methods

| File | Role |
|---|---|
| `method_random.py` | Fits a GP, scores 100k random candidates with UCB and returns the best. |
| `method_grid.py` | Builds a uniform grid over the input space and scores every point with UCB via GP. |
| `method_bayes.py` | Fits a GP, generates Sobol candidates and refines the top ones. Supports global (broad) and tight (local) search modes. |
| `method_surrogate.py` | Replaces the GP with a gradient boosted tree. |
| `method_manual.py` | Debug tool which saves scatter plots for visual inspection. |
| `dispatcher.py` | Routes each function to its method, detects stagnation, adjusts search parameters and logs results. |
| `utils.py` | Shared helpers: Sobol sampling, bound extraction, output formatting, point clipping. |
| `config.py` | All hyperparameters and per-function method assignments. |
| `main.py` | Iterates over all eight functions, loads data, dispatches to the solver and writes a run log. |

---

### Round 1 - Baseline

Used GP regression with random candidate sampling across all functions. 
The GP was fit to the initial data and the candidate with the highest UCB score was submitted. The main goal was to get a first known best value per function before applying anything more refined.

### Round 2 - Bayesian Optimisation with Sobol Candidates

Replaced random sampling with Sobol sequences, which cover the search space more evenly for the same number of points:

- **Sobol sequences** a quasi-random low-discrepancy sequence that spreads points more uniformly across the space than pure random sampling, reducing gaps and clusters for the same number of candidates.
- **Kernel: `ConstantKernel × Matern(ν=2.5)`** the GP covariance function, matern with ν=2.5 assumes the unknown function is smooth but not infinitely differentiable which is a safe assumption for most real-world black-box functions. The ConstantKernel scales the overall output variance.
- **Acquisition:** UCB for low-data or high-dimension cases and EI when more data is available.
- **L-BFGS-B** a gradient-based local optimiser that finds the exact acquisition maximum by iterating from the best Sobol candidates as starting points.
- **Exploration rate:** κ = 2.96 for D ≥ 6, κ = 1.96 otherwise.

### Round 3 - Adaptive Search and Tree-Based Surrogate

Two additions in this round:

1. **Tight search mode** - if a function shows less than 0.5% relative improvement over recent rounds, the search switches from global Sobol sampling to a Gaussian neighbourhood centred on the current best point. This focuses effort locally once a good region has been identified.

2. **Gradient Boosting surrogate** for functions 4, 6, and 8. For higher-dimensional functions with more data, a `GradientBoostingRegressor` (or `RandomForestRegressor`) is fit directly on the labelled data and used to score 16k Sobol candidates. This avoids the GP's scaling issues at higher dimensions and handles less smooth response surfaces better.

The switch between exploration and exploitation is handled automatically by `should_use_tight_search` in `dispatcher.py`, which checks recent improvement history and adjusts the search mode accordingly.