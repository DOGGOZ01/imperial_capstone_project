# Black-Box Optimisation - Imperial Capstone Project

## Abstract

This repository documents a sequential black-box optimisation (BBO) solver built for the BBO Capstone Challenge. The task is to maximise eight unknown scalar functions using only the values returned by an external oracle, one query per function per week, over a period of thirteen weeks. No gradient, formula or internal structure of any function is available at any point in the process. This document describes the solver design, summarises the results after thirteen weeks and links to the supporting analysis, changelog and dataset documentation.

All eight functions improved relative to their initial values, but confidence in the result differs sharply from one function to another, ranging from sharp peaks confirmed over many weeks to functions where the surrogate can barely separate a genuine gain from noise.

---

## Background

Black-box optimisation comes up whenever a function needs to be maximised but its formula is unknown and every evaluation is costly. Examples include tuning the hyperparameters of a machine learning model, screening drug candidates in a laboratory, or adjusting the physical settings of an experiment. In each case the only information available is a record of past evaluations: given an input, what value did the function return.

The standard approach is to fit a statistical model, called a surrogate, to the evaluations collected so far. The surrogate estimates both the expected value of the function at points that have not been tried and the uncertainty of that estimate. An acquisition function combines these two quantities to choose the next point to evaluate, balancing exploration of uncertain regions against exploitation of regions already known to perform well. Once the new evaluation is returned, the surrogate is updated and the process repeats.

This project applies that approach to eight functions with between two and eight input dimensions. Each function represents a different real-world domain: radiation field mapping, a noisy machine learning score, drug discovery, warehouse placement, chemical yield, recipe scoring, machine learning hyperparameter tuning and high-dimensional model performance. Every function is defined over the unit hypercube [0,1]^D, so every input coordinate is a number between 0 and 1. Every submission is one point in that space.

---

## Documentation

### [Results](doc/results.md)
Current best value and best point for each function, with the shape of each function's response surface described using real numbers from all thirteen weeks of data.

### [Rounds](doc/rounds.md)
The full changelog of the solver configuration, week by week, with the reasoning behind each change and the oracle responses that motivated it.

### [Examples](doc/examples.md)
Command output, plot descriptions and an explanation of the difference between `history.json` (solver state) and `oracle_history.json` (the record of submissions).

### [Datasheet](doc/datasheet.md)
Dataset composition, collection process, preprocessing steps and known limitations of the data.

### [Model Card](doc/model_card.md)
A summary of the solver design, its intended use and its known limitations, written in the standard model card format.

---

## Project timeline

Twelve configuration changes were made across thirteen submission weeks (weeks 1 and 2 used the same baseline configuration, so they are counted as one entry in the configuration history but were two separate submissions).

| Week | What changed |
|---|---|
| 1-2 | Baseline: one shared Gaussian Process per function, random candidate points, fixed UCB acquisition, no per-function tuning |
| 3 | Random candidates replaced with Sobol sequences; L-BFGS-B added to refine the acquisition maximum; exploration weight set per dimension |
| 4 | Tight local search introduced for functions that stop improving; tree-based surrogate added for the higher-dimensional functions |
| 5 | Oracle history accumulation fixed, the largest structural bug of the project; full search bounds restored; best point separated from the point being recommended |
| 6 | Log transform introduced for function_1, whose outputs are near zero everywhere; per-function noise and smoothness settings added |
| 7 | Thompson Sampling tried on function_2, the one function with confirmed noisy outputs |
| 8 | Functions 7 and 8 reverted from experimental surrogates back to the Gaussian Process approach; a second bug in the log transform was fixed |
| 9 | A scale-mismatch bug in function_1's progress tracking was fixed, the single most consequential fix of the project up to that point; function_5 reduced to one free dimension; a global refinement method added as an alternative to the default local one |
| 10 | Exploration weight raised for function_7 after its search kept converging to the same region |
| 11 | Alternative search centre activated for function_3; function_1 released from forced global search; new best values for functions 2, 3, 6, 7 and 8 |
| 12 | Periodic global reset removed for function_8; new best values for functions 1, 4, 5, 6 and 8 |
| 13 | function_2 moved back to a standard exploitation-weighted score; function_1's local search window widened; function_1 then improved by a factor of roughly 16,900; new best values also for functions 3, 5, 7 and 8 |

Full week-by-week reasoning: **[doc/rounds.md](doc/rounds.md)**.

---

## Repository layout

| Path | Role |
|---|---|
| `main.py` | Entry point. Loads data for each function, merges in the oracle history, runs the solver and writes `run.txt`. |
| `dispatcher.py` | Decides tight versus global search, the acquisition function and the Gaussian Process settings for each function; updates `history.json`. |
| `config.py` | Every hyperparameter and per-function override: exploration weight, GP noise level, kernel smoothness, tight search radius, global reset interval, acquisition choice, fixed dimensions. |
| `utils.py` | Sobol sampling, point clipping, output string formatting. |
| `logger.py` | Run log with rotation to `run_prev.txt`. |
| `methods/` | `method_bayes.py`, `method_random.py`, `method_grid.py`, `method_surrogate.py`, `method_neural.py`, `method_manual.py`. See [Methods](#methods) below. |
| `data/function_N/` | `initial_inputs.npy` and `initial_outputs.npy` for each function, provided at the start of the challenge. |
| `oracle_history.json` | Every point submitted to the oracle and its response, for all thirteen weeks. |
| `history.json` | Per-function solver state: best value found, the point that achieved it, the point recommended next and counters used to decide the search mode. |
| `plots/` | Diagnostic images produced by the `--manual` mode described in [doc/examples.md](doc/examples.md). |
| `doc/` | [results.md](doc/results.md), [rounds.md](doc/rounds.md), [examples.md](doc/examples.md), [datasheet.md](doc/datasheet.md), [model_card.md](doc/model_card.md). |
| `requirements.txt` | Python dependencies: NumPy, SciPy, scikit-learn, Matplotlib, PyTorch. |
| `LICENSE` | MIT licence covering the code in this repository. The initial datasets under `data/` were provided for the capstone challenge and are not covered by this licence; see [doc/datasheet.md](doc/datasheet.md). |

The search logic itself lives in three files: `dispatcher.py`, `config.py` and `methods/`. Everything else in the table is either input data, state read and written between runs, or generated output. Only `main.py` is meant to be run directly; every other file is imported by it.

---

## The eight functions

| Function | D | Domain |
|---|---|---|
| function_1 | 2 | Radiation field, narrow peak near zero elsewhere |
| function_2 | 2 | Noisy ML model score, stochastic outputs |
| function_3 | 3 | Drug discovery scoring, negative by design |
| function_4 | 4 | Warehouse placement, many local optima |
| function_5 | 4 | Chemical yield, boundary-seeking optimum |
| function_6 | 5 | Recipe scoring, negative by design |
| function_7 | 6 | ML hyperparameter surface |
| function_8 | 8 | High-dimensional ML performance evaluation |

Dimensionality alone spans D=2 to D=8. The domains behind the numbers are different enough (a physical field, a stochastic score, a scoring function that is negative by construction) that no single search configuration performs well across all eight, which is why every function has its own entry in `config.py`.

Each function ships two NumPy arrays, `initial_inputs.npy` (shape `(N, D)`) and `initial_outputs.npy` (shape `(N,)`). A submission is a hyphen-separated string of floats, one per dimension:

```
0.547332-0.244532                              # function_1 (D=2)
0.014749-0.018820-...-0.848244                 # function_8 (D=8)
```

---

## Results

| Function | D | Best value | Best point | Found in |
|---|---|---|---|---|
| function_1 | 2 | 1.13e-06 (raw) / -13.690 (log) | [0.702, 0.686] | week 13 |
| function_2 | 2 | 0.7628 | [0.709, 0.635] | week 11 |
| function_3 | 3 | -0.00690 | [0.524, 0.756, 0.441] | week 13 |
| function_4 | 4 | 0.7307 | [0.366, 0.358, 0.413, 0.416] | week 12 |
| function_5 | 4 | 4463.92 | [0.273, 1.0, 1.0, 1.0] | week 13 |
| function_6 | 5 | -0.2214 | [0.470, 0.370, 0.610, 0.920, 0.069] | week 12 |
| function_7 | 6 | 2.7502 | [0.038, 0.212, 0.447, 0.259, 0.323, 0.749] | week 13 |
| function_8 | 8 | 9.9298 | [0.094, 0.089, 0.108, 0.0, 1.0, 0.429, 0.184, 0.190] | week 13 |

All eight functions improved over their initial dataset value across the thirteen weeks. Week 13 produced the largest change of the whole project: function_1's best output rose from 6.69e-11 to 1.13e-06, a factor of roughly 16,900, at a point in the same small neighbourhood the solver had already been searching. Functions 3, 5, 7 and 8 also reached new best values that week. Functions 2, 4 and 6 held at values found in earlier weeks (11 and 12) and did not improve in week 13. Function_8's result comes with a caveat: all thirteen of its recorded outputs fall within about 6% of each other, so part of its apparent improvement over time may reflect measurement noise rather than a genuinely better input.

Full shape analysis for each function, with the reasoning behind these numbers: **[doc/results.md](doc/results.md)**.

---

## Running it

Install the dependencies once, then run any of the scripts below.

```
pip install -r requirements.txt
```

```
python main.py             # dispatch every function through its configured method, log to run.txt
python main.py --manual    # generate diagnostic plots for every function instead, no method changes
python dimension_checker.py  # quick N / D / max-y printout per function
python reset_history.py      # wipe solver state (best_x, streak, tight_count) back to defaults
```

---

## Methods

`dispatcher.py` reads `history.json` to decide, for each function, whether to search globally over `[0,1]^D` or locally around the current best point, which acquisition function to use and what exploration weight, noise level and kernel smoothness to give the Gaussian Process. Every function currently runs through `method_bayes.py`: a Gaussian Process with a Matern kernel, Sobol candidate sampling and UCB, EI or Thompson Sampling refined by L-BFGS-B or Differential Evolution. Two other surrogates exist for comparison, a tree-based one and a PyTorch model using MC Dropout, alongside the original random and grid baselines. Current per-function settings live in [config.py](config.py); the reasoning behind specific parameter choices over time is in [doc/rounds.md](doc/rounds.md).

### Dispatcher logic (`dispatcher.py`)

Before any method runs, the dispatcher answers three questions using `history.json` and `config.py`.

**Tight or global search?** The dispatcher switches to tight (local) search once a function has gone a set number of weeks without improvement, unless that function is forced into global search or has hit its periodic global-reset interval. As of week 13, every function switches to tight search immediately after any week without improvement.

**Where is the search centred?** The centre defaults to the best point found so far. If a function has gone many weeks without improvement, defined separately for each function, the centre can jump instead to an alternative, previously tried region, on the theory that the current best region is a local trap rather than the true optimum. Function_3 is the only function with this alternative centre defined. It was used in week 11.

**Which acquisition function and exploration weight?** Each function can be given a fixed acquisition function; otherwise the dispatcher falls back to UCB when there is little data or many dimensions and to EI otherwise. The exploration weight is set per function based on how confident the search is in the region it has already found.

For function_5, the dispatcher also removes the three dimensions that were confirmed early on to sit at their upper bound, fits the Gaussian Process on the one remaining free dimension and reconstructs the full four-dimensional point before returning it.

After a method returns a point, the dispatcher compares the new observed value against the stored best value. If it is better, the best value, best point and no-improvement counter are updated; otherwise the counter increases by one. A separate counter tracks consecutive weeks of tight search and triggers a periodic global search once its limit is reached, for the functions that have one configured.

### `method_bayes.py` - Gaussian Process Bayesian Optimisation

The main method, used by every function as of week 13.

1. Fits a Gaussian Process with a Matern kernel on all data collected so far, using noise level and kernel smoothness settings specific to each function. Inputs are used directly since they already live in [0,1]^D.
2. Draws Sobol candidate points: several thousand across the full domain in global mode, or a smaller set inside the local search window in tight mode.
3. In tight mode, the search window is a small box around the current search centre. Its size is set by a shared formula, scaled per function; function_7's window is noticeably larger than function_1's, for example, reflecting how confident the search is in each function's known region.
4. Scores every candidate with the chosen acquisition function: UCB rewards points with high expected value or high uncertainty, EI rewards points likely to beat the current best by a meaningful margin and Thompson Sampling draws one random sample from the model's posterior distribution and picks its highest point directly, without a separate refinement step.
5. For UCB and EI, refines the best candidates using either a gradient-based local optimiser (the default) or a global optimiser called Differential Evolution, used for functions where the gradient-based method kept finding the same point on repeated weeks.
6. Clips the final result to the search window, then to the full [0,1]^D domain.

### `method_random.py` and `method_grid.py` - Baselines

Both methods fit a plain Gaussian Process and score candidates with a fixed UCB acquisition. `method_random` draws 100,000 uniform random candidates; `method_grid` builds a regular grid instead. Neither supports tight search or per-function tuning. These were the original baseline from week 1 and remain in the codebase for comparison, though no function currently uses them.

### `method_surrogate.py` - Tree-Based Surrogate

Replaces the Gaussian Process with a gradient-boosted tree model, or a random forest when there is very little data. This scales better with the number of input dimensions than a Gaussian Process, but produces no uncertainty estimate, so it cannot tell the difference between a region that has not been explored and one that is genuinely poor. This is why the project moved away from it for the higher-dimensional functions once tight search with a Gaussian Process proved more effective.

### `method_neural.py` - MC Dropout MLP

A small neural network trained with dropout left active at prediction time, so that repeated predictions on the same point vary and their spread can be used as an uncertainty estimate, in the same way an ensemble of models would be used. This was tried on the two highest-dimensional functions, since a neural network in principle scales better with dimensionality than a Gaussian Process, but was not kept: both functions performed better once search returned to the Gaussian Process with tight search enabled.

### `method_manual.py` - Diagnostics, Not Optimisation

Triggered by `python main.py --manual`. Produces five image types per function under `plots/`, described with real examples in [doc/examples.md](doc/examples.md). The point it returns is not meant to be submitted; this mode exists only to look at the data.

---

## Conclusion

Thirteen weeks turned one shared configuration into eight separately tuned search strategies. The eight functions ended up split into three groups. Functions 4, 5 and 7 found confident, sharp peaks early and have stayed in permanent local search ever since. Functions 2, 3 and 6 each needed a deliberate move away from a region that looked promising but was not the true optimum; function_3 is the clearest case, its best value roughly tripling the week its search centre moved to a previously untried region. Function_1 and function_8 remain open questions rather than fully solved problems. Function_8's entire recorded range sits within about 6% of itself, narrow enough that the model cannot always tell a real improvement from noise. Function_1 looked, for twelve weeks, like a function with almost no usable signal anywhere in its domain, until week 13 returned a value nearly 17,000 times larger than everything found before it, in a region the search had already been exploring.

That last result is the main lesson of the project. A function that appears flat for a long stretch of a limited evaluation budget is not necessarily flat; it may simply not have been queried precisely enough yet. Deciding that a region, or an entire function, has no accessible signal is a conclusion that should be revisited as more evidence arrives, not treated as settled once a pattern has held for a while. The full week-by-week reasoning behind every configuration change described in this document is in [doc/rounds.md](doc/rounds.md).
