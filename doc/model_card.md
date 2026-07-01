# Model Card: GP-BO Adaptive Solver

## Overview

**Name:** GP-BO Adaptive Solver  
**Type:** Sequential model-based black-box optimisation  
**Repository:** [imperial_capstone_project](../README.md)

A Gaussian Process surrogate fitted to accumulated observations drives a per-function acquisition strategy. The solver selects between global and tight local search based on each function's recent improvement history and applies UCB, EI or Thompson Sampling depending on dimension count and noise characteristics.

---

## Intended Use

**Suitable for:**
- Expensive black-box functions where each evaluation carries a non-trivial cost
- Continuous search spaces bounded to [0,1]^D with D between 2 and 8
- Problems where the budget is severely limited (fewer than 60 total evaluations)
- Functions that are approximately smooth or have isolated peaks

**Not suitable for:**
- Discrete or mixed-integer search spaces
- Functions with more than roughly 20 relevant dimensions (GP training scales cubically with N)
- Multi-objective problems
- Tasks requiring real-time decisions (the solver runs offline once per week)
- Functions with non-stationary behaviour where kernel assumptions break down globally

---

## Details

The solver evolved across thirteen weeks from a single shared configuration to per-function parameter sets covering six categories: GP noise level (alpha), kernel smoothness (nu), UCB exploration coefficient (kappa), tight search radius, global reset interval and acquisition function.

**Week 1-2.** Baseline GP regression with random candidate sampling. One shared configuration across all functions, no per-function tuning.

**Week 3.** Sobol quasi-random sequences replaced random sampling for better space coverage. L-BFGS-B multi-start refinement added for acquisition maximisation.

**Week 4.** Tight search mode introduced: functions with no improvement for two or more consecutive weeks switched to a local neighbourhood centred on the current best. Tree-based surrogate (GradientBoostingRegressor) added for higher-dimensional functions.

**Week 5.** Oracle history accumulation added so the GP trains on all prior submissions rather than only initial data. Search bounds fixed to [0,1]^D. Per-function kappa values introduced. All functions switched to GP-BO.

**Week 6.** Log transform applied to function 1 outputs. Per-function alpha and nu parameters added. Periodic global resets introduced via GLOBAL_SEARCH_INTERVAL. No-improvement streak counter replaced the relative-improvement trigger for tight search activation.

**Week 7.** Thompson Sampling introduced for function 2 (stochastic outputs). Per-function tight search thresholds (TIGHT_STREAK_THRESHOLD) replaced the shared threshold of 2. Function 8 reverted from neural to tree surrogate at small N.

**Week 8.** Functions 7 and 8 reverted from neural and tree surrogates to GP-BO with tight search thresholds set to 0 (always tight). Functions 4 and 5 also set to always-tight mode. Tight search radii reduced for functions 3 and 6 to prevent boundary escape.

**Week 9.** Function 1 returned to forced global search after tight search around [0.728, 0.734] produced no improvement. Log-space bug in history tracking corrected. Function 5 reduced to a 1D GP problem by fixing three confirmed-optimal dimensions. Alternative search centre mechanism added for function 3 (switches to a secondary region after prolonged stagnation). Differential Evolution added as an acquisition optimiser option for functions 1, 7 and 8. Global reset intervals extended and tight radius parameters tightened across several functions.

**Week 10.** Kappa for function 7 raised from 1.5 to 2.0 after DE converged to the same local region on consecutive weeks, requiring broader uncertainty weighting to escape.

**Week 11.** Alternative search centre activated for function 3 after fifteen weeks without improvement in its original region, roughly tripling its best value. Function 1 released from forced global search after ten weeks with no improvement outside its known region. Tight-search threshold set to 0 for every remaining function. Five functions (2, 3, 6, 7 and 8) produced new bests this week.

**Week 12.** Periodic global reset removed for function 8 after reset weeks consistently underperformed the tight-search best. Five functions (1, 4, 5, 6 and 8) produced new bests this week.

**Week 13.** Function 2 moved off Thompson Sampling back to EI as its search window narrowed. Function 1's tight search radius was widened, locking it into permanent tight search. These changes shaped the week 13 submissions, which produced the single largest change of the whole project: function 1 rose from 6.69e-11 to 1.13e-06 (log scale -23.43 to -13.69), a roughly 16,900-fold increase, at a point in the same neighbourhood as the previous best. Functions 3, 5, 7 and 8 also produced new bests this week; functions 2, 4 and 6 did not improve.

---

## Performance

Best values found after all thirteen recorded oracle responses per function. No ground truth is available; values are compared against the initial dataset to assess improvement.

| Function | Best value | Best input (abbreviated) | Improved over initial |
|---|---|---|---|
| function_1 | 1.13e-06 raw / -13.69 log | [0.702, 0.686] | Unknown (log transform complicates comparison) |
| function_2 | 0.7628 | [0.709, 0.635] | Yes |
| function_3 | -0.00690 | [0.524, 0.756, 0.441] | Yes |
| function_4 | 0.7307 | [0.366, 0.358, 0.413, 0.416] | Yes |
| function_5 | 4463.92 | [0.273, 1.0, 1.0, 1.0] | Yes |
| function_6 | -0.2214 | [0.470, 0.370, 0.610, 0.920, 0.069] | Yes |
| function_7 | 2.7502 | [0.038, 0.212, 0.447, 0.259, 0.323, 0.749] | Yes |
| function_8 | 9.9298 | [0.094, 0.089, 0.108, 0.0, 1.0, 0.429, 0.184, 0.190] | Yes |

The optimiser does not track a formal regret metric. Relative improvement over the initial data best and week-on-week progress in `history.json` serve as the primary performance signal. Function 2 provides a lower bound on measurable improvement due to stochastic oracle outputs that make small gains unreliable.

---

## Assumptions and Limitations

**Kernel smoothness assignment.** The Matern nu parameter was assigned per function based on early oracle observations rather than formal model selection. Functions 2, 4 and 8 were assigned nu=1.5 (rough landscape); all others use nu=2.5 (smoother). If this assignment is wrong, the surrogate misrepresents the underlying structure across all subsequent weeks.

**Fixed tight search radius.** The radius 0.15/sqrt(D) scaled by a per-function multiplier from config.py does not adapt based on observed local curvature. Consecutive tight-search queries always sample from the same neighbourhood size regardless of whether that region has been well explored. This can produce near-duplicate queries on flat or noisy functions.

**Stochastic function 2.** The GP treats nearby observations as exact, so two evaluations of the same point returning different values distort the noise model. The alpha=0.05 setting absorbs this partially but does not correct for it. Thompson Sampling reduces the impact by not committing to a single acquisition peak, but the underlying model remains inaccurate.

**GP overconfidence in high dimensions.** For function 8 with 53 observations in 8 dimensions, the GP cannot distinguish a real improvement of 0.01 from measurement noise. All thirteen oracle outputs fell in the range 9.32 to 9.93, a spread of about 6% of the mean value, narrow enough that the surrogate cannot reliably model the landscape and may commit to a suboptimal region without evidence.

**One evaluation per week.** Each week yields exactly one oracle response per function. This means the optimiser cannot verify whether an apparent improvement is real or a noise artefact before committing the next query to the same region. The global reset mechanism partially addresses this, but the reset interval is fixed rather than being based on observed improvement magnitude.

---

## Ethical Considerations

**Transparency and reproducibility.** All submitted points and oracle responses are stored in `oracle_history.json`. The decision of whether each function runs in tight or global mode in any given week is fully deterministic given `config.py`, `history.json` and `oracle_history.json`. A reviewer can reconstruct every recommendation without access to additional notes. Manual parameter choices such as kappa values, nu assignments and radius scales are recorded in `config.py` with inline comments noting when they were changed.

**Real-world relevance.** The functions in this challenge represent real optimisation scenarios where evaluations are expensive: physical experiments, drug trials, simulation runs. The strategy developed here reflects decisions a practitioner would face under similar constraints. The observed failure modes, specifically the risk of premature exploitation on noisy functions and the difficulty of escaping narrow search windows without wasting evaluations, are directly relevant to applied optimisation in science and engineering.

**Scope of conclusions.** Results from this dataset apply only to the eight specific functions in this challenge. The parameter choices (kappa, nu, radius scales) were tuned to these functions over thirteen weeks and are not expected to transfer directly to other problems without re-calibration.
