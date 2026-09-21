"""Evaluate an in-house model through the official `test_unit` harness.

Model names resolve against `src.cp.models.PREDICTORS`; the in-house models
register themselves when their module is imported, so this wrapper imports
them first and then forwards every argument unchanged.

Usage::

    python eval_ours.py --model DD_KOOP_TDD --duplex TDD --test-type regular \
        --limit 0 --out z_artifacts/outputs/repro/CAPNOAUX600_full162.csv
"""

from __future__ import annotations

import sys

import src.cp.models.ours.dd_koop
import src.cp.models.ours.dd_koop_tap
import src.cp.models.baseline.mambacsp
import src.cp.models.baseline.baselines_suite
import src.cp.models.baseline.new_baselines

from repro_official import main

if __name__ == "__main__":
    sys.exit(main())
