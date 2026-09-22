"""Port the DFTGRID baseline into the harness used by the ladder evaluation.

`DFTGRID` (the fixed-grid spectral extrapolator behind the exactness ladder)
lives only in the local working copies; the remote harness that owns the data
does not ship it. This script installs it with the smallest possible edit set and
records the resulting unified diff, because the directive requires any harness
change to be reported verbatim.

Three edits, nothing else:
  1. copy `dftgrid.py` into `src/cp/models/baseline/statistical/`;
  2. import it and register `DFTGRID_TDD` in `src/cp/models/__init__.py`;
  3. add `"DFTGRID"` to `MODELS_NO_CHECKPOINT` in `src/testing/get_models.py`
     (the baseline is parameter free, so it must not look for a checkpoint).

Usage::

    python ladder_install.py --harness <repo> --source <dftgrid.py> --diff <out.diff>
"""

from __future__ import annotations

import argparse
import difflib
import shutil
from pathlib import Path

IMPORT_ANCHOR = "from src.cp.models.baseline.statistical.ar import ARMODEL\n"
IMPORT_LINE = "from src.cp.models.baseline.statistical.dftgrid import DFTGRIDMODEL\n"
REGISTRY_ANCHOR = "    AR_TDD = ARMODEL\n"
REGISTRY_LINE = "    DFTGRID_TDD = DFTGRIDMODEL\n"
NO_CKPT_OLD = 'MODELS_NO_CHECKPOINT = {"NP", "AR", "PAD", "WIENER"}\n'
NO_CKPT_NEW = 'MODELS_NO_CHECKPOINT = {"NP", "AR", "PAD", "WIENER", "DFTGRID"}\n'


def patch(path: Path, anchor: str, addition: str, diff: list) -> None:
    original = path.read_text(encoding="utf-8")
    if addition in original:
        print(f"[install] {path.name}: already patched")
        return
    if anchor not in original:
        raise SystemExit(f"anchor not found in {path}:\n{anchor!r}")
    updated = original.replace(anchor, anchor + addition, 1)
    diff.extend(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=str(path), tofile=str(path)))
    path.write_text(updated, encoding="utf-8")
    print(f"[install] {path.name}: patched")


def replace(path: Path, old: str, new: str, diff: list) -> None:
    original = path.read_text(encoding="utf-8")
    if old not in original:
        print(f"[install] {path.name}: nothing to replace (already updated?)")
        return
    updated = original.replace(old, new, 1)
    diff.extend(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=str(path), tofile=str(path)))
    path.write_text(updated, encoding="utf-8")
    print(f"[install] {path.name}: patched")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    args = parser.parse_args()

    target = args.harness / "src" / "cp" / "models" / "baseline" / "statistical" / "dftgrid.py"
    shutil.copy2(args.source, target)
    print(f"[install] copied {args.source.name} -> {target}")

    diff: list[str] = []
    patch(args.harness / "src" / "cp" / "models" / "__init__.py",
          IMPORT_ANCHOR, IMPORT_LINE, diff)
    patch(args.harness / "src" / "cp" / "models" / "__init__.py",
          REGISTRY_ANCHOR, REGISTRY_LINE, diff)
    replace(args.harness / "src" / "testing" / "get_models.py",
            NO_CKPT_OLD, NO_CKPT_NEW, diff)

    args.diff.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# harness patch for the exactness ladder\n"
              f"# harness: {args.harness}\n"
              f"# new file: {target.relative_to(args.harness)} "
              f"(copied verbatim from {args.source.name})\n\n")
    args.diff.write_text(header + "".join(diff), encoding="utf-8")
    print(f"[install] wrote diff -> {args.diff} ({len(diff)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
