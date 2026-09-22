# ESPRIT per-sample subspace predictor — exploratory, measured negative

Implemented as `src/cp/models/baseline/statistical/esprit.py` (Hankel embedding
per delay tap, forward-backward averaged SVD, ESPRIT shift operator, poles
projected into the unit disk, least-squares gains on the observed window,
continuation of the 32 strongest delay taps, no-prediction for the remainder).

Unit check first: the estimator is exact on synthetic mixtures — a single
on-grid pole, a single off-grid pole (3.5/16) and a two-mode mixture are all
recovered with NMSE 0.0 over the four predicted slots. So the implementation,
not the algebra, is what fails on the benchmark.

Measured through the official harness (`--model ESPRIT --test-type regular`):

| setting | ESPRIT | no-prediction | AR (shipped) | Wiener (shipped) | published |
|---|---|---|---|---|---|
| cm A, 30 ns, 1 m/s, 0 dB | 5.42 / 3.19 / 3.17 / 3.17 | 2.343 | 0.292 | 0.445 | 0.019 |
| cm A, 30 ns, 1 m/s, 25 dB | 3.27 / 1.00 / 1.02 / 1.02 | 0.038 | 0.184 | 0.187 | 0.004 |

(cells show the per-horizon vector where available; NP/AR/Wiener/published are
the shipped rows of `results/reference/official_per_setting.csv`.)

Interpretation: with a single 16-sample window per delay tap, per-sample
subspace estimation is noise-dominated even at 25 dB, and the extrapolation
error does not decay (the poles are projected close to the unit circle). The
classical rungs that *do* compete — AR and Wiener — are statistics-matched:
their coefficients are estimated over the whole training set, which is the
resource a per-sample estimator does not have.

Consequences recorded rather than hidden:
- The paper's classical ladder uses the shipped statistics-matched rows
  (NP / PAD / AR / WIENER) plus the fixed-grid extrapolator (P1); the
  per-sample ESPRIT variant is **not** put in that table, because it is weaker
  than no-prediction and would read as a straw man.
- It is also slow on this path (≈100 s per 162-setting cell on the laptop's
  5060), so a full-grid run was stopped after the two calibration cells.
- A *scenario-matched* super-resolution variant (poles estimated once per
  (channel model, delay spread, speed) from the training set, gains fitted
  per sample) is the defensible way to add this rung; it is left as future
  work and is recorded here so the negative result is not rediscovered.
