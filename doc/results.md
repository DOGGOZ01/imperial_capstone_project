# Results and Function Shapes

Current best value per function, drawn from [history.json](../history.json) after all thirteen weeks of oracle submissions, followed by what the accumulated data in [oracle_history.json](../oracle_history.json) reveals about the shape of each function. For the formal dataset composition see [datasheet.md](datasheet.md); for solver limitations see [model_card.md](model_card.md).

## Current best per function

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

Week 13 produced the single largest jump of the whole project. Function 1's best output rose from 6.69e-11 to 1.13e-06, a factor of roughly 16,900. Functions 3, 5, 7 and 8 also posted new bests that week. Functions 2, 4 and 6 held at values found in earlier weeks (11 and 12) and did not improve in week 13.

---

## function_1 - near-flat with a narrow spike

Domain: radiation field mapping. For twelve weeks, every submitted point returned a value within a few orders of magnitude of zero: 3.97e-86 at [0.648, 0.155], 5.66e-125 at [0.007, 0.340], -1.05e-240 at [0.173, 1.0]. One submission is a clear outlier: [0.422, 0.467] returned -0.00898 in week 2, roughly 230 orders of magnitude larger in absolute value than the rest of the record, which is either a second small feature away from the main peak or measurement noise on a function that is otherwise essentially zero everywhere else it has been probed. Week 13 changed the picture: [0.702, 0.686] returned 1.13e-06, about 16,900 times larger than the previous best of 6.69e-11 found in week 12 at [0.712, 0.718]. Both points sit in the same small neighbourhood (x1 around 0.70-0.71, x2 around 0.69-0.72), so this looks like a genuine local peak coming into focus rather than a jump to a new region. Training happens on `log(y)` because without the transform the GP would see a training set that looks like a flat plane at zero with almost no usable gradient.

## function_2 - noisy, single broad basin

Domain: a noisy ML model score. Range across all thirteen weeks is -0.0005 to 0.7628. The point [0.671, 0.029] was submitted twice, in week 1 and week 3, returning 0.417 then 0.385, a 7.6% relative gap on the same input that confirms genuine output noise rather than a rounding artefact. Away from noise, the trend is a single basin: the further submissions moved toward x1 around 0.7-0.76 and x2 around 0.5-0.7, the higher the return, peaking at 0.7628 at [0.709, 0.635] in week 11. Weeks 12 and 13 stayed in the same neighbourhood (0.659 and 0.663) without beating it.

## function_3 - negative, two basins

Domain: drug discovery scoring, negative by design. The first ten weeks converged on a cluster near [0.95-1.0, 0.94-1.0, 0.39-0.47], which plateaued between -0.178 and -0.0246 without further improvement. Week 11 tried a second region at [0.568, 0.715, 0.441] instead (originally probed back in week 2) and found -0.00852, roughly three times better than the first cluster's plateau. Week 13 refined that same region further, reaching -0.00690 at [0.524, 0.756, 0.441]. The two basins sit far apart in x1 and x2 (0.95 versus 0.52-0.57), so the early convergence on the first cluster was a local optimum rather than the true best.

## function_4 - rugged, many local optima

Domain: warehouse placement. Nearby-looking points swing between strongly negative and positive: [0.520, 0.391, 0.384, 0.373] returns -1.888 while [0.455, 0.395, 0.382, 0.446] returns -0.648, a difference of over 1 despite both points sitting in a similar region. Outputs across all thirteen weeks range from -3.937 to 0.731. Once the search found the right neighbourhood, the best points cluster tightly: all four coordinates of the top few points sit within about 0.06 of each other around x=[0.36-0.42]. The current best, 0.7307 at [0.366, 0.358, 0.413, 0.416], was found in week 12; week 13's attempt at [0.367, 0.340, 0.431, 0.422] returned 0.445, well short of that peak.

## function_5 - one free dimension, still climbing

Domain: chemical yield. Once x2, x3 and x4 were confirmed at their upper bound of 1.0 (week 9), output jumped from the 1000-4000 range into the 4440s and kept rising slowly as x1 increased: 4440.5 at x1=0.000, 4443.3 at x1=0.152, 4452.2 at x1=0.227, 4457.3 at x1=0.250, 4463.9 at x1=0.273 in week 13. That is a small but consistent gain with every increase in x1 across the last five weeks, still positive at the latest sampled point, so the true optimum may sit above 0.273 or the curve may be approaching a plateau just past the sampled range.

## function_6 - negative, boundary-leaning

Domain: recipe scoring, negative by design. Range -0.722 to -0.221 across thirteen weeks. The best cluster sits at x4 close to its upper bound (0.92-1.0 across the strongest submissions) and x5 close to its lower bound (under 0.10), while x1 through x3 vary more freely around [0.40-0.50, 0.31-0.48, 0.52-0.85]. The current best, -0.2214 at [0.470, 0.370, 0.610, 0.920, 0.069], was found in week 12. Week 13 moved to a nearby point and returned -0.374, noticeably worse, suggesting the basin around the week 12 point is narrower than it first appeared.

## function_7 - converged to a narrow corner

Domain: ML hyperparameter surface. Range 0.460 to 2.750 across thirteen weeks (the week 6 submission, at 0.460, is a clear outlier against an otherwise steady climb). Several of the best submissions land at or near x1=0.0, an active boundary rather than an interior optimum, with the rest of the vector settling into a narrow band: x2 around 0.17-0.31, x3 around 0.43-0.58, x4 around 0.16-0.31, x5 around 0.29-0.41, x6 around 0.74-0.90. Week 13 found the current best, 2.7502 at [0.038, 0.212, 0.447, 0.259, 0.323, 0.749], edging x1 slightly away from the boundary while staying inside the same narrow band.

## function_8 - flat at the top, two dimensions pinned

Domain: high-dimensional ML performance evaluation. All thirteen oracle submissions land between 9.32 and 9.93, a spread of about 6% of the mean value across 8 dimensions; the last ten submissions alone span an even narrower 9.83 to 9.93. x4=0.0 and x5=1.0 in the majority of recent submissions, both sitting exactly on their domain boundary. The remaining free dimensions still drift a little between weeks (x6 has ranged 0.41-0.78 across the full history), which the model card flags directly: with output differences this small relative to GP noise assumptions, the surrogate cannot always tell a real improvement from noise. The current best, 9.9298 at [0.094, 0.089, 0.108, 0.0, 1.0, 0.429, 0.184, 0.190], was found in week 13.
