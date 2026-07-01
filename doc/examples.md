# Usage Examples

## Running the solver

```
python main.py
```

Loads `data/function_N/initial_inputs.npy` and `initial_outputs.npy` for each of the 8 functions, merges in every prior submission from `oracle_history.json`, applies the log transform for function_1, then calls `dispatcher.run_method` with the method assigned in `config.py`. Results go to `run.txt`; whatever was in `run.txt` before the run is copied to `run_prev.txt` first, so the previous week's recommendations stay available for comparison.

A real run looks like this:

```
Function     | D   | N     | Method & params                          | Recommended point
------------------------------------------------------------------------------------------------------------------------
function_1   | 2   | 23    | bayes/ei(tight) k=4.00 a=1e-04 nu=2.5 opt=de | 0.712090-0.718090
function_2   | 2   | 23    | bayes/ts(tight) k=2.00 a=5e-02 nu=1.5 opt=lbfgsb | 0.714368-0.695519
function_5   | 4   | 33    | bayes/ei(tight)/free1D k=1.50 a=1e-06 nu=2.5 opt=lbfgsb | 0.250000-1.000000-1.000000-1.000000
function_8   | 8   | 53    | bayes/ucb(tight) k=1.00 a=1e-04 nu=1.5 opt=de | 0.109575-0.072627-0.091607-0.000000-1.000000-0.413314-0.168227-0.174274
```

`N` is the point count after merging initial data with oracle history (function_8 shows 53 because it started with 40 and has 13 recorded oracle weeks). The method column reads as `bayes/<acquisition>(<mode>)[/free<k>D] k=<kappa> a=<gp alpha> nu=<matern nu> opt=<local optimiser>`. `/free1D` on function_5 means the GP trained on 1 free dimension after the other 3 were pinned by `FIXED_DIMS`. The recommended point is the hyphen-separated string ready to submit to the oracle, in the same format the oracle itself expects: `0.547332-0.244532` for a D=2 function.

## Manual diagnostics mode

```
python main.py --manual
```

Runs every function through `method_manual` instead of the configured method for one pass. Nothing in `history.json` changes in a way that affects future tight/global decisions beyond the usual best-value bookkeeping. The point returned is not meant to be submitted. Output lands in `plots/`, five files per function:

- `function_N_scatter.png` - y against each input dimension separately, best point marked with a red star
- `function_N_table.png` - every observed point sorted by y, colour-coded green (best) to red (worst)
- `function_N_parallel.png` - parallel coordinates across all dimensions, line colour and thickness scaled by y rank
- `function_N_heatmap.png` - 2D functions only (function_1, function_2): GP mean and uncertainty over an 80x80 grid
- `function_N_pairplot.png` - 3D and above: every pairwise dimension projection in one scatter matrix

## Other scripts

```
python dimension_checker.py
```

Prints N, D and max observed y for each function directly from the `.npy` files, without touching `oracle_history.json` or running any solver. Useful as a quick sanity check on the raw data shape before a full run.

```
python reset_history.py
```

Overwrites `history.json` with `{'best_value': 0.0, 'was_improved': True}` for all 8 functions, discarding every `best_x`, `no_improvement_streak` and `tight_count`. This is a destructive reset: the next run will treat every function as if it had just improved and start from global search regardless of how far along the search actually was. There is no confirmation prompt, so only run this deliberately.

## How the two history files differ

`oracle_history.json` is an append-only record: every point actually submitted to the oracle and the value it returned, kept per function forever. It is what `main.py` merges into the training data on every run.

`history.json` is solver state, not a data record: `best_value`, `best_x`, `recommended_x`, `tight_count` and `no_improvement_streak` per function, all of it derived and all of it disposable in principle (that is what `reset_history.py` clears). `dispatcher.py` reads and rewrites it every round to decide the next round's search mode.

```json
"function_5": {
  "best_value": 4463.919723736304,
  "best_x": [0.273265, 1.0, 1.0, 1.0],
  "recommended_x": [0.273265, 1.0, 1.0, 1.0],
  "tight_count": 6,
  "no_improvement_streak": 0
}
```

`best_x` is what tight search centres on next round. `recommended_x` is what actually got submitted last round, which can drift from `best_x` since it is the model's suggestion, not a confirmed improvement, until the oracle response proves otherwise.
