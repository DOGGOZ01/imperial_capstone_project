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

### Key Files

| File | Role |
|---|---|
| `method_random.py` | Fits a GP, scores 100k random candidates with UCB and returns the best. |
| `method_grid.py` | Builds a uniform grid over the input space and scores every point with UCB via GP. |
| `method_bayes.py` | Fits a GP, generates Sobol candidates and refines the top ones. Supports global (broad) and tight (local) search modes. Always searches the full [0,1]^D domain. |
| `method_surrogate.py` | Replaces the GP with a gradient boosted tree. Searches the full [0,1]^D domain via Sobol candidates. |
| `method_manual.py` | Debug tool which saves scatter plots for visual inspection. |
| `dispatcher.py` | Routes each function to its method, detects stagnation, adjusts search parameters and logs results. Tracks best observed point separately from the model recommendation. |
| `utils.py` | Shared helpers: Sobol sampling, output formatting, point clipping. |
| `config.py` | All hyperparameters, per-function method assignments and per-function exploration coefficients. |
| `main.py` | Iterates over all eight functions, loads data, merges oracle history, dispatches to the solver and writes a run log. |
| `oracle_history.json` | All points submitted to the oracle and their responses across all weeks. Merged into the training set before each run so the model accumulates knowledge over time. |
| `history.json` | Per-function state used to control search strategy. Stores `best_value`, `best_x` (the point that achieved it), `recommended_x` (the model's next suggestion) and `was_improved`. |

---

### Round 1 - Baseline

Used GP regression with random candidate sampling across all functions. 
The GP was fit to the initial data and the candidate with the highest UCB score was submitted. The main goal was to get a first known best value per function before applying anything more refined.

### Round 2 - Bayesian Optimisation with Sobol Candidates

Replaced random sampling with Sobol sequences, which cover the search space more evenly for the same number of points:

- **Sobol sequences** a quasi-random low-discrepancy sequence that spreads points more uniformly across the space than pure random sampling, reducing gaps and clusters for the same number of candidates.
- **Kernel: `ConstantKernel x Matern(v=2.5)`** the GP covariance function, Matern with v=2.5 assumes the unknown function is smooth but not infinitely differentiable which is a safe assumption for most real-world black-box functions. The ConstantKernel scales the overall output variance.
- **Acquisition:** UCB for low-data or high-dimension cases and EI when more data is available.
- **L-BFGS-B** a gradient-based local optimiser that finds the exact acquisition maximum by iterating from the best Sobol candidates as starting points.
- **Exploration rate:** k = 2.96 for D >= 6, k = 1.96 otherwise.

### Round 3 - Adaptive Search and Tree-Based Surrogate

Two additions in this round:

1. **Tight search mode** - if a function shows less than 0.5% relative improvement over recent rounds, the search switches from global Sobol sampling to a Gaussian neighbourhood centred on the current best point. This focuses effort locally once a good region has been identified.

2. **Gradient Boosting surrogate** for functions 4, 6, and 8. For higher-dimensional functions with more data, a `GradientBoostingRegressor` (or `RandomForestRegressor`) is fit directly on the labelled data and used to score 16k Sobol candidates. This avoids the GP's scaling issues at higher dimensions and handles less smooth response surfaces better.

The switch between exploration and exploitation is handled automatically by `should_use_tight_search` in `dispatcher.py`, which checks recent improvement history and adjusts the search mode accordingly.

### Round 4 - Oracle History, Correct Bounds and Per-Function Tuning

Several structural improvements were made based on analysis of four weeks of results:

**Oracle history accumulation.** Previous rounds rebuilt the GP from only `initial_inputs.npy` each run, discarding all knowledge from past submissions. A new file `oracle_history.json` stores every submitted point and its oracle response. These are merged into the training data in `main.py` before the solver is called, giving the model genuine cumulative learning across weeks.

**Correct search bounds.** The previous implementation derived search bounds from the range of observed data, which artificially restricted exploration to regions already sampled. Both `method_bayes.py` and `method_surrogate.py` now always search the full `[0,1]^D` domain regardless of where existing data falls.

**Separated best point from model recommendation.** `history.json` previously stored only `recommended_x`, which was used as the tight search centre. After each run this was overwritten with the new model suggestion, causing the centre to drift further from the best observed point over time. A new `best_x` field now stores the point that achieved `best_value`. Tight search is always centred on `best_x`, while `recommended_x` records the model's latest suggestion for submission.

**Per-function exploration rates.** A single global k value was replaced with `KAPPA_PER_FUNCTION` in `config.py`:

| Function | k | Reason |
|---|---|---|
| function_1 | 6.0 | All outputs near zero, aggressive exploration needed |
| function_2 | 2.0 | Moderate, improving with higher x2 |
| function_3 | 1.5 | Clear best region found, exploit tightly |
| function_4 | 2.0 | Good region found at centre, refine carefully |
| function_5 | 1.5 | Clear boundary trend, exploit |
| function_6 | 2.5 | Surrogate was degrading, broader search needed |
| function_7 | 1.5 | Best region identified, exploit tightly |
| function_8 | 1.5 | Best pattern found in week 1, preserve and refine |

**All functions switched to Bayesian optimisation.** Functions 4, 6 and 8 previously used tree-based surrogates which do not support tight search. Switching to `method_bayes.py` for all functions ensures that stagnating functions benefit from focused local search rather than continuing to explore randomly.

