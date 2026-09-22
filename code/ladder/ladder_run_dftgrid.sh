#!/bin/bash
# Task A: DFTGRID with a full 16-bin support on the three ladder splits.
#
# Env-driven paths (the as-run machine used absolute locations; see the
# delivery log): LADDER_HARNESS = CSI-4CAST harness checkout, LADDER_OUT =
# scratch/output root. OMP_NUM_THREADS is pinned small on purpose: leaving it
# unset makes torch size its pool from nproc (255) while the cgroup grants far
# fewer CPUs, and the small CPU tensors then spend two orders of magnitude
# more time in thread-pool thrash (measured: 178 s vs 2.3 s per setting).
REPO="${LADDER_HARNESS:?set LADDER_HARNESS to the CSI-4CAST harness checkout}"
cd "$REPO" || exit 1

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export DFTGRID_MODES=16
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

OUT="${LADDER_OUT:-$PWD/ladder_out}/out"
LOG="${LADDER_OUT:-$PWD/ladder_out}"
mkdir -p "$OUT"

# The paper's generalization slice is the 432-setting B/E subset (2 channel
# models x 3 delay spreads x 12 speeds x 6 SNRs). Without these filters
# eval_ours.py sweeps the FULL generalization split instead, and the output
# would not reproduce the paper's ladder row.
GEN_ARGS=(--cm B,E --ds 5e-08,2e-07,4e-07 --ms 3,6,9,12,15,18,21,24,27,33,36,42)

run_split() {           # <test-type> <output-stem> [extra args...]
  local ttype="$1" stem="$2"; shift 2
  python eval_ours.py --model DFTGRID --duplex TDD \
    --test-type "$ttype" --limit 0 "$@" --out "$OUT/$stem.csv" \
    > "$LOG/dftgrid_$stem.log" 2>&1
}

pids=()
run_split regular        DFTGRID_K16_full162   & pids+=($!)
run_split robustness     DFTGRID_K16_robust486 & pids+=($!)
run_split generalization DFTGRID_K16_gen432 "${GEN_ARGS[@]}" & pids+=($!)

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "[dftgrid] split (pid $pid) failed"; fail=1; }
done

# Row-count gate: a short or oversized CSV means the wrong sweep ran.
python - "$OUT" <<'PYEOF'
import csv, pathlib, sys
out = pathlib.Path(sys.argv[1])
want = {"DFTGRID_K16_full162.csv": 162,
        "DFTGRID_K16_robust486.csv": 486,
        "DFTGRID_K16_gen432.csv": 432}
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
    print("[dftgrid] ROW-COUNT GATE FAILED: " + "; ".join(bad))
    sys.exit(1)
print("[dftgrid] row counts OK (162/486/432)")
PYEOF
[ "$?" -ne 0 ] && fail=1

if [ "$fail" -ne 0 ]; then
  echo "[dftgrid] FAILED $(date -u +%H:%M:%S)" >> "$LOG/dftgrid_progress.txt"
  exit 1
fi
echo "[dftgrid] all three splits done $(date -u +%H:%M:%S)" >> "$LOG/dftgrid_progress.txt"
