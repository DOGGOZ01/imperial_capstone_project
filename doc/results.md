# Results and Function Shapes

Current best value per function, drawn from [history.json](../history.json) as of round 12, followed by what the accumulated oracle submissions in [oracle_history.json](../oracle_history.json) reveal about the shape of each function. For the formal dataset composition see [datasheet.md](datasheet.md); for solver limitations see [model_card.md](model_card.md).

## Current best per function

| Function | D | Best value | Best point |
|---|---|---|---|
| function_1 | 2 | 6.69e-11 (raw) / -23.427 (log) | [0.712, 0.718] |
| function_2 | 2 | 0.7628 | [0.709, 0.635] |
| function_3 | 3 | -0.00852 | [0.546, 0.734, 0.463] |
| function_4 | 4 | 0.7307 | [0.366, 0.358, 0.413, 0.416] |
| function_5 | 4 | 4457.29 | [0.25, 1.0, 1.0, 1.0] |
| function_6 | 5 | -0.2214 | [0.470, 0.370, 0.610, 0.920, 0.069] |
| function_7 | 6 | 2.6248 | [0.0, 0.174, 0.434, 0.256, 0.288, 0.745] |
| function_8 | 8 | 9.9193 | [0.110, 0.073, 0.092, 0.0, 1.0, 0.413, 0.168, 0.174] |

---

## function_1 - near-flat with a narrow spike

Domain: radiation field mapping. Almost every submitted point returns a value within a few orders of magnitude of zero: 3.97e-86 at [0.648, 0.155], 5.66e-125 at [0.007, 0.340], -1.05e-240 at [0.173, 1.0]. One submission is a clear outlier: [0.422, 0.467] returned -0.00898, roughly 230 orders of magnitude larger in absolute value than the rest of the record, which is either a second small feature away from the main peak or measurement noise on a function that is otherwise essentially zero everywhere it has been probed. The best value found across all rounds is 6.69e-11 at [0.712, 0.718], which is why training happens on `log(y)` rather than raw y: without the transform the GP sees a training set that looks like a flat plane at zero with no usable gradient.

## function_2 - noisy, single broad basin

Domain: a noisy ML model score. Range across oracle submissions is -0.0005 to 0.7628. The point [0.671, 0.029] was submitted twice and returned 0.417 then 0.385, a 7.6% relative gap on the same input, confirming genuine output noise rather than a rounding artefact. Away from noise, the trend is a single basin: the further submissions moved toward x1 around 0.7-0.76 and x2 around 0.5-0.7, the higher the return, up to the current best 0.7628 at [0.709, 0.635].

## function_3 - negative, two basins

Domain: drug discovery scoring, negative by design. The first eight rounds converged on a cluster near [0.95-1.0, 0.94-1.0, 0.39-0.47], which plateaued between -0.178 and -0.0246 without further improvement. Round 11 tried a second region at [0.568, 0.715, 0.441] instead and found -0.00852, roughly three times better than the first cluster's plateau. The two basins sit far apart in x1 and x2 (0.95 versus 0.57), so the earlier convergence was a local optimum rather than the true best.

## function_4 - rugged, many local optima

Domain: warehouse placement. Nearby-looking points swing between strongly negative and positive: [0.520, 0.391, 0.384, 0.373] returns -1.888 while [0.455, 0.395, 0.382, 0.446] returns -0.648, a difference of over 1 despite both points sitting in a similar region. Outputs across all submissions range from -3.937 to 0.731. Once the search found the right neighbourhood, though, the best points cluster tightly: all four coordinates of the top few points sit within about 0.06 of each other around x=[0.36-0.42].

## function_5 - one free dimension, still climbing

Domain: chemical yield. Once x2, x3 and x4 were confirmed at their upper bound of 1.0 (round 8), output jumped from the 1000-4000 range into the 4440s and kept rising slowly as x1 increased: 4440.5 at x1=0.002, 4440.8 at x1=0.077, 4443.3 at x1=0.152, 4452.2 at x1=0.227, 4457.3 at x1=0.25. That is roughly a 0.4% gain per 0.1 step in x1 over this stretch, still positive at the last sampled point, so the true optimum in x1 may sit above 0.25 or the curve may be approaching a plateau just past the sampled range.

## function_6 - negative, boundary-leaning

Domain: recipe scoring, negative by design. Range -0.722 to -0.221. The best cluster sits at x4 close to its upper bound (0.92-1.0 across the last several submissions) and x5 close to its lower bound (under 0.10), while x1 through x3 vary more freely around [0.40-0.50, 0.31-0.48, 0.52-0.85].

## function_7 - converged to a narrow corner

Domain: ML hyperparameter surface. Range 1.234 to 2.625. Several of the best submissions land at exactly x1=0.0, an active boundary rather than an interior optimum, with the rest of the vector settling into a narrow band: x2 around 0.17-0.31, x3 around 0.43-0.58, x4 around 0.16-0.31, x5 around 0.29-0.41, x6 around 0.74-0.90.

## function_8 - flat at the top, two dimensions pinned

Domain: high-dimensional ML performance evaluation. All nine oracle submissions land between 9.58 and 9.92 (9.32 to 9.91 including the initial dataset per the datasheet), a spread of about 3% of the mean value across 8 dimensions. x4=0.0 and x5=1.0 in five of the last six submissions, both sitting exactly on their domain boundary. The remaining free dimensions still drift a little between rounds (x6 has ranged 0.41-0.78), which the model card flags directly: with output differences this small relative to GP noise assumptions, the surrogate cannot always tell a real improvement from noise.
