"""Count parameters of the published architecture at reduced widths and of the
in-house models, so the capacity-control configurations (`sweep/tdd_o*.yaml`)
can be sized.

Usage (repo root):
    python count_params.py
"""

from __future__ import annotations

import torch

from src.cp.models.proposed.model_tdd import Model
from src.cp.models.ours.dd_koop_tap import DDKoopTapNet
from src.cp.models.ours.dd_koop import DDKoopNet

def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

OFFICIAL = {
    "O50K": dict(dim_model=32, transformer_num_layers=1, transformer_num_heads=4,
                 transformer_hidden_dim=64, embedding_num_res_layers=2,
                 embedding_res_dim=16, embedding_res_groups=2,
                 arl_temporal_proj_num_layers=2, arl_temporal_proj_hidden_dim=32,
                 arl_subcarrier_proj_num_layers=2, arl_subcarrier_proj_hidden_dim=32,
                 denoiser_num_filters_2d=2),
    "O200K": dict(dim_model=64, transformer_num_layers=2, transformer_num_heads=4,
                  transformer_hidden_dim=128, embedding_num_res_layers=2,
                  embedding_res_dim=24, embedding_res_groups=4,
                  arl_temporal_proj_num_layers=2, arl_temporal_proj_hidden_dim=64,
                  arl_subcarrier_proj_num_layers=2, arl_subcarrier_proj_hidden_dim=64,
                  denoiser_num_filters_2d=2),
    "O900K": dict(dim_model=128, transformer_num_layers=3, transformer_num_heads=8,
                  transformer_hidden_dim=256, embedding_num_res_layers=3,
                  embedding_res_dim=32, embedding_res_groups=4,
                  arl_temporal_proj_num_layers=3, arl_temporal_proj_hidden_dim=128,
                  arl_subcarrier_proj_num_layers=2, arl_subcarrier_proj_hidden_dim=128,
                  denoiser_num_filters_2d=3),
    "O2M": dict(dim_model=192, transformer_num_layers=4, transformer_num_heads=8,
                transformer_hidden_dim=384, embedding_num_res_layers=3,
                embedding_res_dim=48, embedding_res_groups=4,
                arl_temporal_proj_num_layers=4, arl_temporal_proj_hidden_dim=192,
                arl_subcarrier_proj_num_layers=2, arl_subcarrier_proj_hidden_dim=192,
                denoiser_num_filters_2d=3),
}

def main() -> None:
    print("--- official architecture at reduced width ---")
    for tag, params in OFFICIAL.items():
        try:
            model = Model(dim_data=600, hist_len=16, pred_len=4, **params)
            print(f"{tag:8s} {count(model):>12,d}")
        except Exception as error:
            print(f"{tag:8s} FAILED: {type(error).__name__}: {error}")

    print("--- ours, already measured ---")
    dk = DDKoopNet(num_subcarriers=300, use_revin=False)
    print(f"{'DK':8s} {count(dk):>12,d}")
    dktap = DDKoopTapNet(num_subcarriers=300, d_model=160, depth=4, use_revin=False)
    print(f"{'DKTAP':8s} {count(dktap):>12,d}")

if __name__ == "__main__":
    main()
