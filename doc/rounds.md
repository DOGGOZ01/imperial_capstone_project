# Round-by-Round Changelog

This file lists what changed in the solver configuration each week, in the order it happened, with the reasoning behind each change. For the current best-known values and function behaviour see [results.md](results.md). For what each module does see the [Methods](../README.md#methods) section in the main README.

---

### Round 1 - Baseline

Used GP regression with random candidate sampling across all functions.
The GP was fit to the initial data and the candidate with the highest UCB score was submitted. The main goal was to get a first known best value per function before applying anything more refined.

### Round 2 - Bayesian Optimisation with Sobol Candidates

Replaced random sampling with Sobol sequences, which cover the search space more evenly for the same number of points.

- **Sobol sequences** are a quasi-random low-discrepancy sequence that spreads points more uniformly across the space than pure random sampling, reducing gaps and clusters for the same number of candidates.
- **Kernel `ConstantKernel x Matern(v=2.5)`** is the GP covariance function. Matern with v=2.5 assumes the unknown function is smooth but not infinitely differentiable, a safe assumption for most real-world black-box functions. The ConstantKernel scales the overall output variance.
- **Acquisition** uses UCB for low-data or high-dimension cases and EI when more data is available.
- **L-BFGS-B** is a gradient-based local optimiser that finds the exact acquisition maximum by iterating from the best Sobol candidates as starting points.
- **Exploration rate** is k = 2.96 for D >= 6 and k = 1.96 otherwise.

### Round 3 - Adaptive Search and Tree-Based Surrogate

Two additions in this round:

1. **Tight search mode** - if a function shows less than 0.5% relative improvement over recent rounds, the search switches from global Sobol sampling to a Gaussian neighbourhood centred on the current best point. This focuses effort locally once a good region has been identified.

2. **Gradient Boosting surrogate** for functions 4, 6 and 8. For higher-dimensional functions with more data, a `GradientBoostingRegressor` (or `RandomForestRegressor`) is fit directly on the labelled data and used to score 16k Sobol candidates. This avoids the GP's scaling issues at higher dimensions and handles less smooth response surfaces better.

The switch between exploration and exploitation is handled automatically by `should_use_tight_search` in `dispatcher.py`, which checks recent improvement history and adjusts the search mode accordingly.

### Round 4 - Oracle History, Correct Bounds and Per-Function Tuning

Several structural improvements were made based on analysis of four weeks of results.

**Oracle history accumulation.** Previous rounds rebuilt the GP from only `initial_inputs.npy` each run, discarding all knowledge from past submissions. A new file `oracle_history.json` stores every submitted point and its oracle response. These are merged into the training data in `main.py` before the solver is called, giving the model genuine cumulative learning across weeks.

**Correct search bounds.** The previous implementation derived search bounds from the range of observed data, which artificially restricted exploration to regions already sampled. Both `method_bayes.py` and `method_surrogate.py` now always search the full `[0,1]^D` domain regardless of where existing data falls.

**Separated best point from model recommendation.** `history.json` previously stored only `recommended_x`, used as the tight search centre. After each run this was overwritten with the new model suggestion, causing the centre to drift further from the best observed point over time. A new `best_x` field now stores the point that achieved `best_value`. Tight search is always centred on `best_x`, while `recommended_x` records the model's latest suggestion for submission.

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

### Round 5 - Neural Surrogate, Per-Function GP Tuning and Log Transform

Several targeted improvements based on analysis of four weeks of accumulated data.

**Neural surrogate for functions 7 and 8.** A new `method_neural.py` module introduces a PyTorch MLP with MC Dropout as a surrogate model. The network is a three-hidden-layer architecture (64 to 64 to 32 to 1) with 15% dropout. After training, dropout is kept active and 50 forward passes are run over Sobol candidates to estimate both mean and uncertainty. UCB is then applied to these estimates. MC Dropout is a practical approximation of Bayesian inference in neural networks. Keeping dropout active at inference time causes the network to behave as an ensemble, giving a cheap uncertainty estimate without requiring a GP. Functions 7 (D=6) and 8 (D=8) are the most complex and have the most accumulated data, making them good candidates for a higher-capacity model. If PyTorch is not installed, the method falls back to `method_surrogate`.

**Log transform for function_1.** Function 1 outputs are near-zero across almost the entire domain, with a narrow spike at the true maximum. A GP fit directly on these near-zero values has very little signal to work with. `main.py` now applies a log transform to function_1 outputs before fitting (`log(y)` for positive values, `-300` for non-positive). This spreads the response surface and makes the peak much more visible to the GP.

**Richer per-function GP configuration.** Four new dictionaries were added to `config.py` to give each function its own GP hyperparameters:

| Parameter | Purpose |
|---|---|
| `GP_ALPHA_PER_FUNCTION` | Per-function GP noise level (alpha). Higher for noisy functions (f2), lower for deterministic ones. |
| `MATERN_NU_PER_FUNCTION` | Matern smoothness: `nu=2.5` for smooth unimodal landscapes, `nu=1.5` for rough multi-modal ones. |
| `N_CANDIDATES_PER_FUNCTION` | Override candidate count per function. Function_1 uses 8192 for finer coverage of its narrow peak. |
| `TIGHT_RADIUS_SCALE` | Multiplier on the tight search radius `0.15/sqrt(D)`. Functions 3 and 4 use 0.6 to focus more tightly. |

**Periodic global resets.** A new `GLOBAL_SEARCH_INTERVAL` dict in `config.py` controls how many consecutive tight searches are allowed before forcing a global reset. Function_2 resets every 3 tight iterations (many local peaks) and function_3 every 4 (risk of corner trap). This is tracked via `tight_count` in `history.json`.

**Improved stagnation detection.** The previous tight-search trigger was based on a relative improvement ratio, which was sensitive to scale. It is now replaced with a simple `no_improvement_streak` counter that activates tight search after 2 consecutive rounds without improvement. The counter is reset to zero whenever a new best is found and is stored in `history.json` alongside `tight_count`.

**GP scaler removed from `method_bayes`.** The `MinMaxScaler` that was applied to GP inputs has been removed. Since all inputs already live in `[0, 1]^D` by construction, the scaler was a no-op that added complexity without benefit. The GP now fits and predicts directly on the raw `[0,1]` values and the L-BFGS-B optimiser uses `[(0,1)]^D` bounds directly.

**Tight search candidates switched to Sobol.** Previously, tight search used Gaussian noise around the centre, which could cluster candidates near the mean. It now uses Sobol samples shifted and scaled to the tight window, giving better uniform coverage of the local neighbourhood.

### Round 6 - Thompson Sampling, Per-Function Acquisition and Surrogate Reversion

Changes based on analysis of Round 5 results, where functions 2, 4 and 6 regressed due to a first-run initialisation issue that sent them into global search mode before the `no_improvement_streak` counter was populated.

**Thompson Sampling for function_2.** A new acquisition option `acq_func='ts'` was added to `method_bayes.py`. Thompson Sampling draws one sample from the GP posterior using `gp.sample_y()` and returns the argmax over candidates. Unlike UCB or EI, it requires no kappa to tune and naturally balances exploration and exploitation through posterior uncertainty. It is particularly well suited to stochastic or multimodal functions where UCB/EI can get overconfident. The L-BFGS-B optimisation loop is skipped entirely for TS: the candidate argmax is the result. Function_2 (a noisy ML model with many local peaks) is the first function assigned TS via the new `ACQ_FUNC_PER_FUNCTION` dict in `config.py`.

**Per-function acquisition override.** `ACQ_FUNC_PER_FUNCTION` in `config.py` allows any function to bypass the default `ucb`/`ei` selection logic in `adaptive_bayes_params` and use a fixed acquisition function. This is cleaner than adding special cases to the dispatcher.

**Per-function tight-search threshold.** The hardcoded `no_improvement_streak >= 2` trigger was replaced with a `TIGHT_STREAK_THRESHOLD` dict (default 2). Function_2 is set to 4 because its stochastic outputs mean two rounds without improvement is weak evidence for a true plateau, so more patience before switching to local search is warranted.

**Function_8 reverted to GBDT surrogate.** Function_8 produced only a marginal +0.17% improvement with the neural surrogate in Round 5. At N=46 data points, a gradient-boosted tree is typically more stable than an MLP. Trees do not overfit as aggressively at small N and do not require hyperparameter tuning of the training procedure. Function_7 stays on the neural surrogate since it showed a meaningful +8.9% improvement.

**Tight-search bounds safety clip.** After the L-BFGS-B optimisation loop, the result is now explicitly clipped to the tight window bounds. This prevents rare numerical edge cases where the optimiser exits the tight region by a small margin.

**GP kernel diagnostics.** A `verbose` flag and `_log_kernel()` helper were added to `method_bayes.py`. When enabled, the optimised kernel parameters (constant scale, per-dimension length scales and log marginal likelihood) are printed after each GP fit. A warning is emitted if any length scale is at its optimisation boundary, indicating the GP kernel may be misspecified for that function.

**Kappa raised for function_2.** Kappa was increased from 1.5 to 2.0 to give broader exploration for the noisy landscape. This applies when TS is not active, for example during a forced global reset round.

### Round 7 - Surrogate Reversion, Tight-Search Hardening and Log Transform Fix

Changes driven by analysis of Round 6 results, where functions 7 and 8 regressed under their alternative surrogates and several functions continued to waste evaluations on global search despite well-identified best regions.

**Functions 7 and 8 reverted to GP-BO.** Function_7's MC Dropout neural surrogate produced a -28% regression at N=43. The MLP is too data-hungry and unstable at this sample size. Function_8's GBDT surrogate consistently ignored the known best pattern (x4 near 0, x5 near 1) because global Sobol search spreads evaluations uniformly rather than concentrating on the known good region. Both functions are now back on `method_bayes` with tight search enabled immediately via `TIGHT_STREAK_THRESHOLD=0`. Function_7 exploits around [0.071, 0.198, 0.544, 0.268, 0.354, 0.803] and function_8 exploits around [0.094, 0.025, 0.108, 0.0, 1.0, 0.461, 0.120, 0.127].

**Functions 4 and 5 added to `TIGHT_STREAK_THRESHOLD=0`.** Function_4's peak at [0.373, 0.401, 0.409, 0.403] is very sharp and any global round wastes the evaluation. Function_5 confirmed that x2=x3=x4=1 is optimal and only x1 needs refinement. After the Round 6 improvement reset the streak to 0, the next run was incorrectly reverting to global search and recommending [1,1,1,1]. Setting threshold=0 keeps both functions permanently in tight mode regardless of the improvement streak.

**Tight search radius reduced for functions 3 and 6.** Function_3's EI acquisition was repeatedly pushing to the x1=1.0 and x2=1.0 boundary regardless of tight radius, because the gradient always pointed toward the corner. The radius scale was reduced from 0.6 to 0.25 (physical radius approximately 0.022) to force the search to stay within a narrower window around the known best. Function_6's tight search was consistently escaping the best-known region; its radius scale was reduced from 1.0 to 0.4.

**`GLOBAL_SEARCH_INTERVAL` extended to functions 4 and 6.** Function_4 now forces a global reset every 10 tight rounds as a safety valve, preventing the search window from drifting too far from the true peak indefinitely. Function_6, with `tight_count=7` at round start and no prior reset path, now resets every 8 tight rounds.

**Function_1 removed from `FORCE_GLOBAL`.** After seven rounds of forced global search across the full 2D domain, no point outside [0.728, 0.734] produced a meaningful signal. The function is confirmed to have a single narrow spike. Function_1 now uses tight search centred on [0.728, 0.734] with `kappa=6.0` for local exploration. Kappa is kept high because the peak is very narrow and the surrounding landscape is essentially flat: broad local uncertainty estimates are needed to find the spike centre.

**Log transform outlier fix for function_1.** The previous implementation mapped non-positive outputs to a fixed value of -300. Since all other log-transformed values in the dataset fall in approximately [-33, -34], this created a single point approximately 265 standard deviations below the mean, which severely distorted the GP's normalisation. Non-positive outputs are now mapped to `finite_min - 10`, where `finite_min` is the minimum finite log value in the dataset. This keeps all training values on a consistent scale.

**GP optimiser restarts reduced from 15 to 8.** Profiling showed that 15 restarts added significant runtime without meaningfully changing the optimised kernel parameters. Eight restarts are sufficient for the kernel hyperparameter landscape encountered in these functions.

### Round 8 - Global Search for Function 1, Dimensionality Reduction for Function 5 and Differential Evolution

Changes driven by analysis of oracle results across all eight rounds, by identifying a log-space bug in function_1 history tracking and by observing that the tight search for function_5 could be simplified to a 1D problem.

**Function_1 returned to global search.** In Round 7, function_1 was moved into tight search around [0.728, 0.734] on the assumption that the true maximum was in that region. After comparing results with other approaches, it became clear that all eight oracle submissions returned near-zero values and that the tight region was almost certainly not near the true maximum. Function_1 is now back in `FORCE_GLOBAL` with `kappa=4.0` to encourage exploration of the full domain. The tight search radius setting is left in config for reference but has no effect while FORCE_GLOBAL is active.

**Log-space bug fixed for function_1.** Since Round 5, `main.py` applies a log transform to function_1 outputs before passing them to the solver. The GP and `update_history` therefore work with log-scale values, which are large negative numbers. However, `history.json` still stored `best_value = 1.587e-15` from the original scale, set before the log transform was introduced. The comparison `data_best_y > prev_best` always failed because log-scale values (around -34) are numerically less than 1.587e-15, so `history.json` never updated for function_1 after Round 5. The stored value has been corrected to -34.076, which is `log(1.587e-15)`. `no_improvement_streak` was also reset so the history could update correctly from this round onward.

**Function_5 reduced to a 1D problem.** Across eight rounds, x2, x3 and x4 were consistently confirmed at 1.0 for the best observed outputs. Rather than fitting a 4D GP that must model three effectively constant dimensions, a new `FIXED_DIMS` dictionary in `config.py` specifies which dimensions are pinned to fixed values. The dispatcher slices the training matrix to the free dimensions, fits the GP on that reduced input space and reconstructs the full-dimensional recommendation before returning. For function_5 this means the GP trains on x1 alone, substantially improving surrogate accuracy and acquisition function precision for refining a single variable.

**Alternative search centre for function_3.** Function_3 has not improved since round three. Rather than triggering a full global reset after prolonged stagnation, which risks discarding the region around the best observed point, a new `ALTERNATIVE_CENTER` mechanism shifts the tight search to the second best observed point [0.568, 0.715, 0.441] once `no_improvement_streak` exceeds a configurable threshold. This preserves local search discipline while escaping the current trapped region.

**Differential Evolution added as acquisition function optimiser.** `method_bayes.py` now accepts a `local_optimizer` parameter. The default remains `lbfgsb` (L-BFGS-B multi-start). Setting it to `de` uses `scipy.optimize.differential_evolution` instead, which searches the acquisition function globally over the given bounds rather than from multiple local starting points. DE is less sensitive to the initial population and does not require gradient information, making it better suited to multimodal or high-dimensional acquisition surfaces. The `LOCAL_OPT_PER_FUNCTION` dictionary in `config.py` assigns DE to function_1 (global search, unknown landscape) and functions 7 and 8 (high-dimensional tight search where L-BFGS-B consistently found the same local optimum).

**Tight search parameters tightened across several functions.** Based on analysis of eight rounds of results, the following radius and kappa values were revised:

| Function | Change | Reason |
|---|---|---|
| function_2 | Radius 1.5 to 0.5 | Thompson Sampling was jumping far from best [0.676, 0.537] |
| function_4 | Radius 0.6 to 0.3 | Sharper focus around confirmed best [0.373, 0.401, 0.409, 0.403] |
| function_6 | Added to TIGHT_STREAK_THRESHOLD=0 | Exploit new best -0.234 immediately after global reset |
| function_7 | Radius 1.0 to 0.8 | Slight reduction to keep search closer to best |
| function_8 | Kappa 2.5 to 1.0, radius 1.0 to 0.3 | x6 was consistently drifting away from known optimal value 0.461 |

**Global search intervals extended.** To prevent premature global resets that waste evaluations on functions with well-identified best regions, `GLOBAL_SEARCH_INTERVAL` was updated:

| Function | Old interval | New interval |
|---|---|---|
| function_2 | 3 | 8 |
| function_3 | 4 | 8 |
| function_4 | 10 | 20 |

**Visual diagnostics added to `method_manual.py`.** The manual mode previously saved only per-dimension scatter plots. It now generates three additional images per function:

- `_table.png` - all observed points sorted by output value, colour-coded from best to worst.
- `_parallel.png` - parallel coordinates plot with lines coloured by output rank, showing which input combinations produced high outputs.
- `_heatmap.png` (2D functions only) - GP mean prediction and uncertainty on an 80x80 grid, with observed points overlaid.
- `_pairplot.png` (3D and above) - scatter matrix of all pairwise dimension projections, with the best observed point marked.

**`--manual` flag added to `main.py`.** Running `python main.py --manual` switches all functions to `method_manual` for one run, generating the full set of diagnostic images without modifying the submission recommendations. Running `python main.py` without the flag uses the configured method per function as before.

### Round 9 and 10 - Kappa Adjustment for Function 7 and Global Reset Removal for Function 8

Changes based on analysis of Round 8 results, where the Differential Evolution acquisition optimiser and the 1D dimensionality reduction for function 5 produced the first sustained improvement on that function. Tight search on functions 6 and 7 kept converging toward new bests over the same period.

**Kappa raised for function 7.** After Round 8, DE converged to the same local region in the 6D search space on consecutive tight-search rounds. Kappa was raised from 1.5 to 2.0 to increase the uncertainty bonus in the UCB acquisition function and encourage exploration of slightly different areas within the tight window on each round.

**Global reset interval removed for function 8.** The periodic global reset for function 8 was removed after reset rounds consistently produced outputs near 9.3, well below the tight search best of 9.90. At 49 observations across 8 dimensions, a global Sobol search spreads candidates too thinly to outperform the known good region, making the reset counterproductive.

Three functions found new bests from this round's submissions. Function 5 reached 4443.3 at [0.152, 1.0, 1.0, 1.0], confirming the upward trend in the first dimension under the 1D GP. Function 6 reached -0.234 at [0.417, 0.358, 0.610, 0.973, 0.074], the best result on that function across all rounds. Function 7 reached 2.566 at [0.010, 0.223, 0.483, 0.207, 0.329, 0.742].

### Round 11 - Alternative Centre Trigger, Function 1 Tight Search and Final Config Hardening

**Function 3 alternative centre activated.** After fifteen rounds without improvement in the [0.966, 0.941, 0.448] region, the `ALTERNATIVE_CENTER_STREAK` threshold was reached and tight search shifted to [0.568, 0.715, 0.441]. The oracle returned -0.00852, the best result on this function across all rounds and roughly three times better than the previous best of -0.0246. This confirmed that the earlier cluster was a local optimum and the alternative region contains a meaningfully better basin.

**Function 1 removed from `FORCE_GLOBAL`.** After ten rounds of global search with no oracle query returning a value better than the initial dataset point at [0.728, 0.734], function_1 was removed from `FORCE_GLOBAL`. The final round uses tight search centred on [0.728, 0.734] to refine near the only known high-output region.

**`TIGHT_STREAK_THRESHOLD` set to 0 for all remaining functions.** Functions 3, 6 and 7 previously had default thresholds that allowed one or two global rounds after a new best was found. Setting all thresholds to 0 ensures every function enters tight search immediately, with no wasted evaluations on global rounds after the best regions are confirmed.

Five functions improved this round. Function 2 reached 0.7628 at [0.709, 0.635]. Function 3 reached -0.00852 from its newly activated alternative centre, as described above. Function 6 reached -0.2225 at [0.444, 0.355, 0.637, 0.946, 0.069]. Function 7 reached 2.6248 at [0.0, 0.174, 0.434, 0.256, 0.288, 0.745]. Function 8 reached 9.9169 at [0.125, 0.057, 0.108, 0.0, 1.0, 0.429, 0.152, 0.158].

### Round 12 - Function 2 Off Thompson Sampling, Function 1 Locked Into Tight Search

**Function_2 switched from Thompson Sampling back to EI.** `ACQ_FUNC_PER_FUNCTION` dropped `function_2: 'ts'` in favour of `'ei'`, alongside the same override for functions 1, 7 and 8. TS had been in place since Round 6 for its resistance to overconfidence on noisy outputs, but with `no_improvement_streak` now at 5 for function_2 and the last several tight rounds landing within a tight cluster around [0.71, 0.64], EI's exploitation bias suits the narrowed search window better than a fresh posterior draw every round.

**Function_1 added to `TIGHT_STREAK_THRESHOLD=0` and its tight radius widened.** `TIGHT_RADIUS_SCALE['function_1']` moved from 0.15 to 0.3, doubling the local search window around the best point. Function_1's signal is so close to flat that a narrow radius risks sampling noise indistinguishable from the peak. Combined with `TIGHT_STREAK_THRESHOLD=0`, function_1 now stays in tight mode permanently rather than alternating with global rounds.

**New bests from this round's submissions.** Function_1 improved from -34.076 to -23.42711 on the log scale (raw output about 6.69e-11) at [0.71209, 0.71809], still within the same narrow neighbourhood as before. Function_4 jumped from 0.5597 to 0.7306640198500962 at [0.366364, 0.358133, 0.413088, 0.415806], a roughly 30% gain and the largest single-round improvement recorded for that function up to this point. Function_5 reached 4457.2925 at [0.25, 1.0, 1.0, 1.0], continuing its steady climb in the free dimension. Function_6 edged up from -0.2225 to -0.2214023578798504 at [0.470474, 0.370138, 0.609772, 0.919501, 0.069183], a marginal refinement of the same basin found in Round 9. Function_8 reached 9.9192502179124 at [0.109575, 0.072627, 0.091607, 0.0, 1.0, 0.413314, 0.168227, 0.174274].

### Week 13 - Largest Single-Week Change of the Project

The config from Round 12 (function_2 back to EI, function_1's tight radius widened) generated the week 13 submissions. The results were the most significant of the whole project.

**Function_1 jumped roughly 16,900-fold.** The oracle returned 1.1338e-06 at [0.701999, 0.686270], compared to the previous best of 6.6945e-11 at [0.71209, 0.71809] found in Round 12. Both points sit in the same neighbourhood (x1 near 0.70-0.71, x2 near 0.69-0.72), so this reads as the tight search finally resolving the true peak rather than a jump to a different region. On the log scale used for training, this moves the best value from -23.427 to -13.690.

**Three more functions improved.** Function_3 reached -0.006896049688374402 at [0.524353, 0.755634, 0.440917], a further refinement of the alternative centre found in Round 11. Function_5 reached 4463.919723736304 at [0.273265, 1.0, 1.0, 1.0], continuing its climb in the free dimension. Function_7 reached 2.750158545240856 at [0.038351, 0.211799, 0.446783, 0.258855, 0.323400, 0.748524]. Function_8 reached 9.9298181522144 at [0.093665, 0.088537, 0.107517, 0.0, 1.0, 0.429224, 0.184137, 0.190184].

**Three functions did not improve.** Function_2 returned 0.6625781624422553, function_4 returned 0.4448996362390045 and function_6 returned -0.3743545226655401, all below their respective best values. No configuration change has yet been made in response to these results; that will be Round 13's task.

Current config and history state is in [config.py](../config.py) and [history.json](../history.json). See [results.md](results.md) for the full best-value table and per-function shape analysis.
