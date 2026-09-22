"""Install the LSGAINS baseline into the harness (second half of the port).

Same three-edit pattern as `ladder_install.py`, and the diff is appended to the
same patch file so the delivery carries one complete record of every harness
change made for the ladder.

Usage::

    python ladder_install_lsgains.py --harness <repo> --source ladder_lsgains.py \
        --diff <ladder_harness_patch.diff>
"""

from __future__ import annotations

import argparse
import difflib
import shutil
from pathlib import Path

IMPORT_ANCHOR = "from src.cp.models.baseline.statistical.dftgrid import DFTGRIDMODEL\n"
IMPORT_LINE = "from src.cp.models.baseline.statistical.lsgains import LSGAINSMODEL\n"
REGISTRY_ANCHOR = "    DFTGRID_TDD = DFTGRIDMODEL\n"
REGISTRY_LINE = "    LSGAINS_TDD = LSGAINSMODEL\n"
NO_CKPT_OLD = 'MODELS_NO_CHECKPOINT = {"NP", "AR", "PAD", "WIENER", "DFTGRID"}\n'
NO_CKPT_NEW = 'MODELS_NO_CHECKPOINT = {"NP", "AR", "PAD", "WIENER", "DFTGRID", "LSGAINS"}\n'


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
        print(f"[install] {path.name}: nothing to replace")
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

    target = (args.harness / "src" / "cp" / "models" / "baseline" / "statistical"
              / "lsgains.py")
    shutil.copy2(args.source, target)
    print(f"[install] copied {args.source.name} -> {target}")

    diff: list[str] = []
    patch(args.harness / "src" / "cp" / "models" / "__init__.py",
          IMPORT_ANCHOR, IMPORT_LINE, diff)
    patch(args.harness / "src" / "cp" / "models" / "__init__.py",
          REGISTRY_ANCHOR, REGISTRY_LINE, diff)
    replace(args.harness / "src" / "testing" / "get_models.py",
            NO_CKPT_OLD, NO_CKPT_NEW, diff)

    with open(args.diff, "a", encoding="utf-8") as handle:
        handle.write(f"\n# new file: {target.relative_to(args.harness)} "
                     f"(copied verbatim from {args.source.name})\n\n")
        handle.write("".join(diff))
    print(f"[install] appended diff -> {args.diff} ({len(diff)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
