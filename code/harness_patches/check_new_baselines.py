"""Parameter count and forward-shape check for the new baseline family.

Run from a repo root with the harness importable. Prints one line per model so
the sizes can be quoted in the comparison table before anything is trained.
"""

from __future__ import annotations

import torch

from src.cp.models.baseline.new_baselines import (
    ITransformerNet,
    MambaLiteNet,
    TimeMixerNet,
    TSMixerNet,
)

CASES = [
    ("TimeMixer-S (ICLR24)", TimeMixerNet, dict(d_model=32, scales=(1, 2, 4), kernel=3)),
    ("TimeMixer-M", TimeMixerNet, dict(d_model=96, scales=(1, 2, 4, 8), kernel=3)),
    ("TimeMixer-L", TimeMixerNet, dict(d_model=192, scales=(1, 2, 4, 8), kernel=3)),
    ("TSMixer-S (2023)", TSMixerNet,
     dict(hidden=64, feature_hidden=64, groups=40, blocks=2)),
    ("TSMixer-M", TSMixerNet, dict(hidden=128, feature_hidden=256, groups=40, blocks=2)),
    ("TSMixer-L", TSMixerNet, dict(hidden=256, feature_hidden=1024, groups=40, blocks=3)),
    ("iTransformer-S (ICLR24)", ITransformerNet,
     dict(groups=60, d_model=32, heads=4, layers=2)),
    ("iTransformer-M", ITransformerNet, dict(groups=100, d_model=64, heads=4, layers=3)),
    ("iTransformer-L", ITransformerNet, dict(groups=120, d_model=96, heads=4, layers=4)),
    ("Mamba-lite-S", MambaLiteNet, dict(d_model=16, d_state=16, expand=2, kernel=4)),
    ("Mamba-lite-M", MambaLiteNet, dict(d_model=24, d_state=16, expand=2, kernel=4)),
    ("Mamba-lite-L", MambaLiteNet, dict(d_model=32, d_state=16, expand=2, kernel=4)),
    ("PatchTST matched (reference)", None, {}),
]

def main() -> int:
    if CASES[-1][1] is None:
        from src.cp.models.baseline.baselines_suite import PatchTSTNet

        CASES[-1] = ("PatchTST matched (reference)", PatchTSTNet,
                     dict(patch_len=4, d_model=16, layers=2, heads=4))
    x = torch.randn(2, 16, 600)
    for label, network, params in CASES:
        model = network(hist_len=16, pred_len=4, channels=600, **params)
        total = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            y = model(x)
        finite = bool(torch.isfinite(y).all())
        print(f"{label:32s} params={total:>9,} out={tuple(y.shape)} finite={finite}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
