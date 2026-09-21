# Code

The experiment code, unchanged. Two things are needed to run it: the official
CSI-4CAST repository (the benchmark harness) and the CSI-4CAST datasets from the
project's Hugging Face organisation.

## What is here

| path | contents |
|---|---|
| `harness_patches/` | the project's own driver and wrappers, copied from the machine that ran them: `train_sweep.py` (patched, see below), `train_ours.py`, `eval_ours.py`, `profile_models.py`, `repro_official.py`, `check_new_baselines.py`, `make_new_baseline_configs.py`, `summarize_csv.py`, `count_params.py` |
| `models_ours/` | `dd_koop.py` - the reported architecture (registers `DD_KOOP_TDD`) - and `dd_koop_tap.py`, a tap-encoding variant |
| `models_baseline/` | `baselines_suite.py` (DLinear, MLP, GRU, TCN, Transformer, PatchTST), `new_baselines.py` (TimeMixer, TSMixer, iTransformer, Mamba-lite), `mambacsp.py` (2026 competitor) |
| `config/sweep/` | the 60 training configs used by the reported runs |
| `analysis/` | `regenerate_all.py` (one entry point: every derived table in `results/analysis/`, dependency order), `audit_claims.py` (number-to-source audit) and the individual table/figure drivers |
| `figures/` | the intervention evaluator `eval_pole_offset.py` (grid snap / quantisation / top-k) and the figure scripts |

## Install into a working copy of the harness

```bash
git clone <CSI-4CAST repository>                 # the benchmark harness
cp harness_patches/*.py            <repo>/       # train/eval drivers
cp models_ours/*.py                <repo>/src/cp/models/ours/
cp models_baseline/*.py            <repo>/src/cp/models/baseline/
cp -r config/sweep                 <repo>/z_artifacts/config/ours/sweep/
```

Then place the datasets at `<repo>/z_artifacts/data/{train,test}/regular/...` and
the normalisation statistics at `<repo>/z_artifacts/data/stats/`.

## Entry scripts

| task | command (from the repo root) |
|---|---|
| train the reported configuration | `python train_sweep.py z_artifacts/config/ours/sweep/tdd_caplong.yaml` |
| train a battery in one packing pass | `python train_sweep.py <cfg1> <cfg2> ... --summary summary.json` |
| override a knob without editing yaml | append `::TAG::key=value,key=value`, e.g. `z_artifacts/config/ours/sweep/tdd_caplong.yaml::CAPNOAUX600::auxi_lambda=0.0` |
| score one checkpoint on 162 settings | `python eval_ours.py --model DD_KOOP_TDD --duplex TDD --test-type regular --limit 0 --out out.csv` |
| score the robustness split | same with `--test-type robustness` |
| score the generalization slice | same with `--test-type generalization` plus `--cm/--ds/--ms` per combination (the harness selects single triples) |
| reproduce the published model | `python repro_official.py --model MODEL --duplex TDD --test-type regular --limit 0 --out out.csv` |
| profile parameters/MACs/latency | `python profile_local.py <configs...>` |
| rebuild the handover tables | `python analysis/build_tables.py` (run from the handover root) |

## Two patches to the harness that the reported numbers depend on

`train_sweep.py` in `harness_patches/` differs from the upstream copy in exactly
two places, both of which the final runs use:

1. `_apply_override` understands `seed=` (so a multi-seed battery can be trained
   from one packing pass) and `batch=` (so a screening job can halve its
   activation memory).
2. It imports `src.cp.models.baseline.new_baselines` so the recent-baseline
   family is registered.

`eval_ours.py` additionally imports `new_baselines`; without that import the
harness raises "not registered in PREDICTORS" and writes no rows
(the shipped `eval_ours.py` copy already carries it).

## Environment notes that cost time

- The container memory cap is 100 GB; one full-protocol packing pass holds
  ~43 GB, so two concurrent full-protocol jobs do not fit but one plus a
  9-subset screen does.
- Three concurrent jobs on one 24 GB card die of CUDA OOM; two fit at batch 16.
- On Windows the DataLoader must stay in the main process (`CSI_WORKERS=0`), and
  the worker count in `train_ours.py` has to be pinned to 0 as well.
- The remote hosts have no outbound internet; data was fetched locally through a
  mirror and pushed over `scp`.

## Software environment

- Analysis stack (regenerates every table/figure from the shipped CSVs):
  Python 3.10.9, numpy 1.26.4, pandas 2.3.3, PyYAML 6.0.3, matplotlib 3.10.9,
  PyTorch Lightning 2.6.6; any stable `torch>=2.1` works (CPU suffices). See
  `requirements-lock.txt`.
- Latency profiles: NVIDIA RTX 4090 D, driver 580.119.02, fp32, batch 32
  (`results/analysis/profiles_4090.txt` records the protocol).
- Loading the released checkpoints with torch >= 2.6 requires
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the shipped eval scripts set it).
- Deployment note: on the tested stack the complex-valued core runs fp32
  only; half precision and `torch.compile` are unsupported for this
  operator. All efficiency numbers in the paper are fp32 predictor-level.

## One-command regeneration of every table and figure

```bash
python code/analysis/regenerate_all.py             # 22 analysis steps (~35 s)
python code/analysis/regenerate_all.py --figures   # + the eight figures (~2 min)
```

`regenerate_all.py` runs the analysis scripts in dependency order. Every script
is pure: it reads `results/run_csv/*.csv` (the raw per-setting truth) plus
`results/reference/official_per_setting.csv` (the benchmark's published table)
and writes into `results/analysis/`. Run twice and the derived tables are
byte-identical - the per-run logs under `results/analysis/` are rewritten
from scratch on every run - which is the idempotence check used before
hand-over.

The figure step also runs `paper/figures/flatten_rasters.py`, which writes opaque-RGB
600 dpi rasters. If a long-lived process holds a memory-mapped section on a
canonical `.tiff`, the flattener writes `<name>.rgb.tiff` next to it (verified
RGB) and reports that explicitly rather than leaving a translucent file.

### Statistics conventions (identical in every table)

* cells are `(channel model, delay spread, speed)`; NMSE is averaged over the
  SNR levels inside a cell, then over the cells. The robustness split uses the
  same cell definition, *not* one cell per corruption type - adding
  `noise_type` to the cell key rescales the table (0.162 -> 0.142) and must not
  be done;
* paired comparisons use 5{,}000 bootstrap resamples with RNG `20260913`,
  plus an exact sign test `max(favour, n - favour)`;
* all numbers are TDD-only. The benchmark's reference table carries both TDD
  and FDD rows for channel models A/C/D; mixing them inverts the sign of the
  A/C/D shift result;
* MACs count matmul and convolution only; the parameter-free DFT/IDFT and the
  elementwise products are excluded.

### Paper artefact → command → inputs (traceability map)

| paper artefact | produced by | key inputs |
|---|---|---|
| Table 1 (main comparison) | `python code/analysis/build_tables.py` | `results/run_csv/*_full162.csv`, `results/reference/official_per_setting.csv` |
| Table 1 classical rows | `code/analysis/classical_ladder_extend.py` via `build_tables.py` | `results/reference/official_per_setting.csv`, `results/run_csv/DFTGRID_K1_*.csv` |
| Table 1 capacity rows | `build_tables.py` (MAIN list) | `results/run_csv/CAPLONG_CAPACITY_*.csv` |
| §IV-B capacity scan | `code/analysis/capacity_curve.py` via `build_tables.py` | same capacity run CSVs |
| §IV-C three-seed shift | `code/analysis/generalization_seeds.py` | `results/run_csv/{CAPNOAUX600,CAPNOAUXS43,HEADNOAUXS44}_gen_gen.csv` |
| §IV-C speed bands | `code/analysis/generalization_by_band.py` | same three run CSVs |
| §IV-C interpolation axis (A/C/D) | `code/analysis/acd_three_seeds.py` | `results/run_csv/*_gen_acd.csv` |
| §IV-C controls on the 432 slice | `code/analysis/generalization_432_summary.py` | `results/run_csv/{P3CONV,MAMBA64CONV_C,TRANSMATCHED}_gen*.csv` |
| §IV-D loss-off 2³ factorial | `build_tables.py` (ABLOFF list) | `results/run_csv/ABLOFF_*_full162.csv` |
| §IV-D pole-grid intervention | `code/analysis/pole_intervention_analysis.py` | `results/run_csv/{SNAPPOLE_*,POLEQ_*,POLETOP*}_full162.csv` |
| §IV-D intervention on shift slices | `code/analysis/slice_summary_extended.py` | `results/run_csv/{SNAP0,XS_snap0,_gen_BE,_gen_ACD,_robust486}.csv` |
| §IV-D intervention vs capacity | `code/analysis/pole_snap_capacity.py` | `results/run_csv/{TINY_stock,TINY_snap0,CAPLONG_CAPACITY_XS,XS_snap0}_full162.csv` |
| §IV-E latency/MACs | `results/analysis/profiles_4090.txt`, `code/analysis/profile_local.py` | forward passes on the evaluation GPU |
| §IV-E MRT rate comparison | `code/figures/deploy_profile.py` + `rate_eval.py` (harness side) | `results/analysis/rate_comparison.csv` |
| Fig. 2 (`fig_pareto`) | `python paper/figures/make_fig_pareto.py` | `results/analysis/main_table.csv`, `profiles_4090.txt` |
| Fig. 3 (`fig_condition`) | `python paper/figures/make_fig_condition.py` | `results/analysis/generalization_by_condition*.csv` |
| supplementary mechanism figures | `python paper/figures/make_fig_pole.py`, `make_fig_pole_gain.py`, `make_fig_mobility.py`, `make_fig_anatomy.py`, `make_fig_snap_slices.py` | `results/analysis/pole_export.csv`, `pole_intervention_curve.csv`, `generalization_seeds.csv`, `scene_dump_*.npz` |
| audit (every quoted number) | `python code/analysis/audit_claims.py` | `paper/main.tex` + all of the above |

### Four self-checks worth running before any release

1. Idempotency (every derived table byte-identical on the second run):

```
python code/analysis/regenerate_all.py && cp -r results/analysis /tmp/snap1
python code/analysis/regenerate_all.py && diff -rq /tmp/snap1 results/analysis
# the diff must print nothing
```

2. Package self-sufficiency (every `results/analysis/*.csv` source reference
   resolves to a file that ships in this repository):

```python
import re, pathlib, collections
ROOT = pathlib.Path(".")
ref = re.compile(r"results/[A-Za-z0-9_./\-]+\.csv")
miss = collections.Counter()
for p in (ROOT / "results" / "analysis").glob("*.csv"):
    for m in set(ref.findall(p.read_text(encoding="utf-8", errors="ignore"))):
        if not (ROOT / m.rstrip(';').rstrip(',')).exists():
            miss[m] += 1
print("missing:", len(miss))
```

3. Figure pipeline (every figure script must run in the released layout):

```
python code/analysis/regenerate_all.py --figures
# exit code must be 0; [ok ] = rebuilt, [skip] = an input that ships outside this
# package (the scene dumps behind fig_anatomy), [FAIL] = a path or an input broke
```

4. The shipped PDF must match the LaTeX source (the numbers a reviewer reads
   live in the PDF, not in the .tex):

```
pdftotext paper/main.pdf - | grep -c "0.1367"     # must be >= 1
pdftotext paper/main.pdf - | grep -c "0.1383"     # must be 0
pdfinfo paper/main.pdf | grep Pages               # 5 pages (4 + references)
```

`audit_claims.py` must run from the repository root (it reads
`paper/main.tex` next to `code/` and `results/`); it exits with a clear
error if the layout is missing.
