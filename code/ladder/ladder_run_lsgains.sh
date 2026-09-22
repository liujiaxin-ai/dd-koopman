#!/bin/bash
# Task B: evaluate the ridge-LS grid gains as a standalone analytic predictor.
#
# Seven processes: the native warm-start fit and the full-training fit on all
# three splits, plus the analytic periodic-extension gains on `regular`, the
# cross-check that must reproduce the DFTGRID K=16 harness number.
# OMP_NUM_THREADS is pinned small on purpose (see ladder_run_dftgrid.sh):
# seven processes share a small CPU quota, and an unpinned pool makes each
# process spawn hundreds of threads (measured 2.3 s -> 178 s per setting).
REPO="${LADDER_HARNESS:?set LADDER_HARNESS to the CSI-4CAST harness checkout}"
cd "$REPO" || exit 1

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"

GAINS="${LADDER_GAINS:?set LADDER_GAINS to the directory holding W_*.npz}"
OUT="${LADDER_OUT:-$PWD/ladder_out}/out"
LOG="${LADDER_OUT:-$PWD/ladder_out}"
mkdir -p "$OUT"

run() {                 # <W-path> <stem> <test-type> [extra args...]
  local w="$1" stem="$2" ttype="$3"; shift 3
  LSGAINS_W_PATH="$w" python eval_ours.py --model LSGAINS \
    --duplex TDD --test-type "$ttype" --limit 0 "$@" --out "$OUT/$stem.csv" \
    > "$LOG/lsgains_$stem.log" 2>&1
}

# Same 432-setting B/E filter as ladder_run_dftgrid.sh -- without it the
# generalization runs sweep the full split and cannot reproduce the paper.
GEN_ARGS=(--cm B,E --ds 5e-08,2e-07,4e-07 --ms 3,6,9,12,15,18,21,24,27,33,36,42)

pids=()
run "$GAINS/W_native.npz" LSGAINS_K16_full162 regular & pids+=($!)
run "$GAINS/W_native.npz" LSGAINS_K16_robust486 robustness & pids+=($!)
run "$GAINS/W_native.npz" LSGAINS_K16_gen432 generalization "${GEN_ARGS[@]}" & pids+=($!)
run "$GAINS/W_full.npz" LSGAINSFULL_K16_full162 regular & pids+=($!)
run "$GAINS/W_full.npz" LSGAINSFULL_K16_robust486 robustness & pids+=($!)
run "$GAINS/W_full.npz" LSGAINSFULL_K16_gen432 generalization "${GEN_ARGS[@]}" & pids+=($!)
run "$GAINS/W_periodic.npz" LSGAINSPER_full162 regular & pids+=($!)

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "[lsgains] run (pid $pid) failed"; fail=1; }
done

# Row-count gate over all seven expected outputs.
python - "$OUT" <<'PYEOF'
import csv, pathlib, sys
out = pathlib.Path(sys.argv[1])
want = {"LSGAINS_K16_full162.csv": 162, "LSGAINS_K16_robust486.csv": 486,
        "LSGAINS_K16_gen432.csv": 432, "LSGAINSFULL_K16_full162.csv": 162,
        "LSGAINSFULL_K16_robust486.csv": 486, "LSGAINSFULL_K16_gen432.csv": 432,
        "LSGAINSPER_full162.csv": 162}
bad = []
for name, n in want.items():
    p = out / name
    try:
        rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    except FileNotFoundError:
        bad.append("%s missing" % name)
        continue
    if len(rows) != n:
        bad.append("%s: %d rows (want %d)" % (name, len(rows), n))
if bad:
    print("[lsgains] ROW-COUNT GATE FAILED: " + "; ".join(bad))
    sys.exit(1)
print("[lsgains] row counts OK (162/486/432 x2 + periodic 162)")
PYEOF
[ "$?" -ne 0 ] && fail=1

if [ "$fail" -ne 0 ]; then
  echo "[lsgains] FAILED $(date -u +%H:%M:%S)" >> "$LOG/lsgains_progress.txt"
  exit 1
fi
echo "[lsgains] all seven runs done $(date -u +%H:%M:%S)" >> "$LOG/lsgains_progress.txt"
