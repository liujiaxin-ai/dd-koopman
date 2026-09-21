# CSI-4CAST official baseline table (rebuilt from shipped result.csv)

Per-setting NMSE averaged over the 4 prediction horizons. `MODEL` is the
paper's proposed architecture; the rest are baselines.

| model | regular | robustness | generalization | all splits | settings |
|---|---|---|---|---|---|
| AR | 0.7874 | 0.7692 | 1.0434 | 0.9963 | 3708 |
| CNN | 0.4446 | 0.4229 | 0.5370 | 0.5180 | 7416 |
| LLM4CP | 0.2278 | 0.2437 | 0.4728 | 0.4321 | 7416 |
| MODEL | 0.1969 | 0.2192 | 0.4663 | 0.4221 | 7416 |
| NP | 2.2724 | 1.7678 | 2.4511 | 2.3537 | 7416 |
| PAD | 0.2300 | 0.6052 | 0.3354 | 0.3662 | 3708 |
| RNN | 0.2218 | 0.2431 | 0.4734 | 0.4322 | 7416 |
| STEMGNN | 0.3726 | 0.3683 | 0.4942 | 0.4724 | 7416 |
| WIENER | 0.9366 | 0.8841 | 1.0904 | 1.0566 | 7416 |

## Per duplex

| key | settings | mean NMSE |
|---|---|---|
| AR|TDD|generalization | 3060 | 1.0434 |
| AR|TDD|regular | 162 | 0.7874 |
| AR|TDD|robustness | 486 | 0.7692 |
| CNN|FDD|generalization | 3060 | 0.7447 |
| CNN|FDD|regular | 162 | 0.6799 |
| CNN|FDD|robustness | 486 | 0.6418 |
| CNN|TDD|generalization | 3060 | 0.3293 |
| CNN|TDD|regular | 162 | 0.2093 |
| CNN|TDD|robustness | 486 | 0.2040 |
| LLM4CP|FDD|generalization | 3060 | 0.6838 |
| LLM4CP|FDD|regular | 162 | 0.3143 |
| LLM4CP|FDD|robustness | 486 | 0.3077 |
| LLM4CP|TDD|generalization | 3060 | 0.2619 |
| LLM4CP|TDD|regular | 162 | 0.1413 |
| LLM4CP|TDD|robustness | 486 | 0.1797 |
| MODEL|FDD|generalization | 3060 | 0.6728 |
| MODEL|FDD|regular | 162 | 0.2692 |
| MODEL|FDD|robustness | 486 | 0.2762 |
| MODEL|TDD|generalization | 3060 | 0.2598 |
| MODEL|TDD|regular | 162 | 0.1246 |
| MODEL|TDD|robustness | 486 | 0.1623 |
| NP|FDD|generalization | 3060 | 2.6818 |
| NP|FDD|regular | 162 | 2.7367 |
| NP|FDD|robustness | 486 | 2.2280 |
| NP|TDD|generalization | 3060 | 2.2205 |
| NP|TDD|regular | 162 | 1.8081 |
| NP|TDD|robustness | 486 | 1.3076 |
| PAD|TDD|generalization | 3060 | 0.3354 |
| PAD|TDD|regular | 162 | 0.2300 |
| PAD|TDD|robustness | 486 | 0.6052 |
| RNN|FDD|generalization | 3060 | 0.6393 |
| RNN|FDD|regular | 162 | 0.2796 |
| RNN|FDD|robustness | 486 | 0.2893 |
| RNN|TDD|generalization | 3060 | 0.3075 |
| RNN|TDD|regular | 162 | 0.1640 |
| RNN|TDD|robustness | 486 | 0.1969 |
| STEMGNN|FDD|generalization | 3060 | 0.6464 |
| STEMGNN|FDD|regular | 162 | 0.5243 |
| STEMGNN|FDD|robustness | 486 | 0.5212 |
| STEMGNN|TDD|generalization | 3060 | 0.3421 |
| STEMGNN|TDD|regular | 162 | 0.2209 |
| STEMGNN|TDD|robustness | 486 | 0.2153 |
| WIENER|FDD|generalization | 3060 | 1.0946 |
| WIENER|FDD|regular | 162 | 1.0497 |
| WIENER|FDD|robustness | 486 | 0.9948 |
| WIENER|TDD|generalization | 3060 | 1.0861 |
| WIENER|TDD|regular | 162 | 0.8236 |
| WIENER|TDD|robustness | 486 | 0.7734 |
