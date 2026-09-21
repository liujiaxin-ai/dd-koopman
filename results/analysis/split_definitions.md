# Split definitions (X1 / X2)

Everything below is read off the benchmark harness code and the `ms` column of
the run CSVs; no definition is inferred from prose.

## Where the axes are defined

| element | file:line | value |
|---|---|---|
| regular channel models | `src/utils/data_utils.py:104` | `["A", "C", "D"]` |
| regular delay spreads | `src/utils/data_utils.py:105` | `[30, 100, 300] ns` |
| regular speeds | `src/utils/data_utils.py:107` | `[1, 10, 30] m/s` |
| generalization channel models | `src/utils/data_utils.py:114` | `["A", "B", "C", "D", "E"]` |
| generalization delay spreads | `src/utils/data_utils.py:115` | `[30, 50, 100, 200, 300, 400] ns` |
| generalization speeds | `src/utils/data_utils.py:116` | `3..45 step 3, plus 1 and 10` |
| vanilla SNR grid | `src/noise/noise_testing.py:56` | `[0, 5, 10, 15, 20, 25] dB` |
| phase-noise SNR grid | `src/noise/noise_testing.py:57` | `[10, 15, 20, 25] dB` |
| burst-noise SNR grid | `src/noise/noise_testing.py:58` | `[10, 15, 20, 25] dB` |
| package-drop rates | `src/noise/noise_testing.py:61` | `[0.01 … 0.10]` (10 values) |

## regular — 162 settings

`3 channel models × 3 delay spreads × 3 speeds = 27 scenarios`, each evaluated
at the six vanilla SNRs: `27 × 6 = 162`. Training uses the same 27 scenarios
(`train_ratio = 0.9`).

## robustness — 486 settings (X1)

Built by `create_robustness_combinations` (`src/testing/config.py:129`):

| noise type | levels | scenarios | settings |
|---|---|---|---|
| phase | SNR `10, 15, 20, 25` dB | 27 | 108 |
| burst | SNR `10, 15, 20, 25` dB | 27 | 108 |
| package drop | rate `0.01 … 0.10` | 27 | 270 |
| **total** | | | **486** |

The scenario grid is the regular one (channel models A/C/D, delay spreads
30/100/300 ns, speeds 1/10/30 m/s); only the noise process changes.

## generalization — 3060 settings

`5 channel models × 6 delay spreads × 17 speeds = 510 scenarios`, six vanilla
SNRs each: `510 × 6 = 3060`. Channel models B and E, delay spreads
50/200/400 ns and all speeds except 1/10 m/s are absent from training.

## The 216- and 432-setting sub-slices used in the paper (X2)

Both slices share the grid `channel models {B, E} × delay spreads {50, 200, 400} ns`
and exclude the two in-distribution speeds 1 and 10 m/s.

| slice | speeds (m/s) | cells | settings | where it appears |
|---|---|---|---|---|
| 432 | 3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 36, 42 | 72 | 432 | `CAPNOAUX600_gen_gen.csv`, `LONGFULL72_gen.csv` |
| 216 | 3, 9, 15, 21, 27, 33 | 36 | 216 | `MAMBA64_gen.csv`, `TRANSMATCHED_gen.csv`, `LONGFULL_gen.csv` |

Selection rule, stated honestly: the 216 slice is **the subset that fitted the
6.5 GB staging budget** in `fetch_generalization.py --budget-gb 6.5`, whose
`SPEEDS` constant is `(3, 9, 15, 21, 27, 33)` — every second speed of the wider
slice. It is a budget artefact, not a principled subset; the 432 slice is the
wider download with the remaining even speeds (6, 12, 18, 24, 36, 42) added.
The reference table covers both slices (it carries the full 510-scenario grid).

Evidence: the `ms` column of the CSVs named above. The 432 runs contain 72
cells, the 216 runs 36, and both are subsets of the reference grid.
