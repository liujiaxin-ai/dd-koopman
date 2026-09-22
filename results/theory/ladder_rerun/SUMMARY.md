# Ladder re-run verification (fixed runner, 2026-09-22)

Source: `results/theory/ladder_rerun/` (never overwrites `results/run_csv/`).

| split | rows (want) | rows (got) | cell-mean (re-run) | cell-mean (expected) | vs shipped CSV |
|---|---|---|---|---|---|
| regular | 162 | 162 | **3.2883** | 3.2883 | max|Δ|=2.000e-06 over 162 matched settings |
| robustness | 486 | 486 | **2.7558** | 2.7558 | max|Δ|=1.225e-06 over 486 matched settings |
| generalization-432 | 432 | 432 | **2.7518** | 2.7518 | max|Δ|=6.750e-07 over 432 matched settings |

## gen432 structure (the point of the fix)

channel models = ['B', 'E'] (exactly B and E, no A/C/D), delay spreads = [5e-08, 2e-07, 4e-07], speeds = [3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42] (12 values)

* (cm, ds, ms) cells = **72** = 2 x 3 x 12; rows per cell = 6..6 (the 6 SNRs) -> total **432**
* rows per channel model = 216 = 3 ds x 12 speeds x 6 SNR
* rows per (cm, ds) = 72 = 12 speeds x 6 SNR

| cm | ds | speeds | rows | rows/speed |
|---|---|---|---|---|
| B | 5e-08 | 12 | 72 | [6] |
| B | 2e-07 | 12 | 72 | [6] |
| B | 4e-07 | 12 | 72 | [6] |
| E | 5e-08 | 12 | 72 | [6] |
| E | 2e-07 | 12 | 72 | [6] |
| E | 4e-07 | 12 | 72 | [6] |

Full (cm, ds, ms) cell table:

| cm | ds | ms | rows |
|---|---|---|---|
| B | 5e-08 | 3 | 6 |
| B | 5e-08 | 6 | 6 |
| B | 5e-08 | 9 | 6 |
| B | 5e-08 | 12 | 6 |
| B | 5e-08 | 15 | 6 |
| B | 5e-08 | 18 | 6 |
| B | 5e-08 | 21 | 6 |
| B | 5e-08 | 24 | 6 |
| B | 5e-08 | 27 | 6 |
| B | 5e-08 | 33 | 6 |
| B | 5e-08 | 36 | 6 |
| B | 5e-08 | 42 | 6 |
| B | 2e-07 | 3 | 6 |
| B | 2e-07 | 6 | 6 |
| B | 2e-07 | 9 | 6 |
| B | 2e-07 | 12 | 6 |
| B | 2e-07 | 15 | 6 |
| B | 2e-07 | 18 | 6 |
| B | 2e-07 | 21 | 6 |
| B | 2e-07 | 24 | 6 |
| B | 2e-07 | 27 | 6 |
| B | 2e-07 | 33 | 6 |
| B | 2e-07 | 36 | 6 |
| B | 2e-07 | 42 | 6 |
| B | 4e-07 | 3 | 6 |
| B | 4e-07 | 6 | 6 |
| B | 4e-07 | 9 | 6 |
| B | 4e-07 | 12 | 6 |
| B | 4e-07 | 15 | 6 |
| B | 4e-07 | 18 | 6 |
| B | 4e-07 | 21 | 6 |
| B | 4e-07 | 24 | 6 |
| B | 4e-07 | 27 | 6 |
| B | 4e-07 | 33 | 6 |
| B | 4e-07 | 36 | 6 |
| B | 4e-07 | 42 | 6 |
| E | 5e-08 | 3 | 6 |
| E | 5e-08 | 6 | 6 |
| E | 5e-08 | 9 | 6 |
| E | 5e-08 | 12 | 6 |
| E | 5e-08 | 15 | 6 |
| E | 5e-08 | 18 | 6 |
| E | 5e-08 | 21 | 6 |
| E | 5e-08 | 24 | 6 |
| E | 5e-08 | 27 | 6 |
| E | 5e-08 | 33 | 6 |
| E | 5e-08 | 36 | 6 |
| E | 5e-08 | 42 | 6 |
| E | 2e-07 | 3 | 6 |
| E | 2e-07 | 6 | 6 |
| E | 2e-07 | 9 | 6 |
| E | 2e-07 | 12 | 6 |
| E | 2e-07 | 15 | 6 |
| E | 2e-07 | 18 | 6 |
| E | 2e-07 | 21 | 6 |
| E | 2e-07 | 24 | 6 |
| E | 2e-07 | 27 | 6 |
| E | 2e-07 | 33 | 6 |
| E | 2e-07 | 36 | 6 |
| E | 2e-07 | 42 | 6 |
| E | 4e-07 | 3 | 6 |
| E | 4e-07 | 6 | 6 |
| E | 4e-07 | 9 | 6 |
| E | 4e-07 | 12 | 6 |
| E | 4e-07 | 15 | 6 |
| E | 4e-07 | 18 | 6 |
| E | 4e-07 | 21 | 6 |
| E | 4e-07 | 24 | 6 |
| E | 4e-07 | 27 | 6 |
| E | 4e-07 | 33 | 6 |
| E | 4e-07 | 36 | 6 |
| E | 4e-07 | 42 | 6 |