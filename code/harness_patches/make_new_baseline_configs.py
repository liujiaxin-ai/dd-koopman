"""Emit configurations for the four recent lightweight baselines.

Companion to `new_baselines.py`. Each configuration is the same training recipe
the rest of the comparison family gets (batch 32, Adam at 5e-4, ReduceLROnPlateau,
300 epochs with patience 25, 27-subset regular training protocol), so a
difference between two rows is attributable to the architecture and not to the
schedule. Widths are chosen to bracket the reported model's 173,190 parameters
instead of dwarfing it, because the claim under test is a parameter/accuracy
trade.

Usage (remote, repo root, venv python):
    python make_new_baseline_configs.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from make_tap_sweep import BASE

COMMON = dict(hist_len=16, pred_len=4, channels=600)

MODELS = {
    "TIMEMIXERS": dict(name="TIMEMIXER_TDD",
                       params={**COMMON, "d_model": 32, "scales": [1, 2, 4], "kernel": 3}),
    "TIMEMIXERM": dict(name="TIMEMIXER_TDD",
                       params={**COMMON, "d_model": 96, "scales": [1, 2, 4, 8], "kernel": 3}),
    "TIMEMIXERL": dict(name="TIMEMIXER_TDD",
                       params={**COMMON, "d_model": 192, "scales": [1, 2, 4, 8], "kernel": 3}),
    "TSMIXERS": dict(name="TSMIXER_TDD",
                     params={**COMMON, "hidden": 64, "feature_hidden": 64, "groups": 40, "blocks": 2}),
    "TSMIXERM": dict(name="TSMIXER_TDD",
                     params={**COMMON, "hidden": 128, "feature_hidden": 256, "groups": 40, "blocks": 2}),
    "TSMIXERL": dict(name="TSMIXER_TDD",
                     params={**COMMON, "hidden": 256, "feature_hidden": 1024, "groups": 40, "blocks": 3}),
    "ITRANSFORMERS": dict(name="ITRANSFORMER_TDD",
                          params={**COMMON, "groups": 60, "d_model": 32, "heads": 4, "layers": 2}),
    "ITRANSFORMERM": dict(name="ITRANSFORMER_TDD",
                          params={**COMMON, "groups": 100, "d_model": 64, "heads": 4, "layers": 3}),
    "ITRANSFORMERL": dict(name="ITRANSFORMER_TDD",
                          params={**COMMON, "groups": 120, "d_model": 96, "heads": 4, "layers": 4}),

    "MAMBALITES": dict(name="MAMBA_LITE_TDD",
                       params={**COMMON, "d_model": 16, "d_state": 16, "expand": 2, "kernel": 4}),
    "MAMBALITEM": dict(name="MAMBA_LITE_TDD",
                       params={**COMMON, "d_model": 24, "d_state": 16, "expand": 2, "kernel": 4}),
    "MAMBALITEL": dict(name="MAMBA_LITE_TDD",
                       params={**COMMON, "d_model": 32, "d_state": 16, "expand": 2, "kernel": 4}),
}

def main() -> None:
    target = Path("z_artifacts/config/ours/sweep")
    target.mkdir(parents=True, exist_ok=True)
    for tag, spec in MODELS.items():
        config = dict(BASE)
        config["experiment_name"] = f"TDD_{tag}"
        config["model_name"] = tag
        config["output_dir"] = f"z_artifacts/outputs/sweep/{tag}"
        config["model"] = {
            "checkpoint_path": None,
            "is_separate_antennas": True,
            "name": spec["name"],
            "params": spec["params"],
        }
        path = target / f"tdd_new_{tag.lower()}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=True)
        print(f"wrote {path}")

if __name__ == "__main__":
    main()
