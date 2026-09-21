# P3 - published architecture at 91,410 parameters, trained to plateau or 6 h

| field | value |
|---|---|
| tmux/session | none available on this host; launched with setsid + nohup |
| training log | /root/p3_conv.log |
| supervisor log | /root/p3_finish.log |
| started (UTC) | 2026-09-14T16:10:10Z |
| finished (UTC) | 2026-09-14T22:13:28Z |
| run directory | `z_artifacts/outputs/sweep/O93K_P3CONV/2026-09-14_16-19-43/` |
| best checkpoint | `epoch=077-step=059280-val_loss=0.167139.ckpt` |
| best validation NMSE | 0.167139 |
| checkpoints written | 1 |
| official 162-setting NMSE | 0.1807 over 162 settings |
| evaluation CSV | z_artifacts/outputs/repro/P3CONV_full162.csv |
| tensorboard | z_artifacts/outputs/sweep/O93K_P3CONV/2026-09-14_16-19-43/tb_logs/20260914161944/events.out.tfevents.1789402785.jupyter-c3a9je5a9webtaf1.3918.0 |

Previous number for this control: 0.2174 at 40 epochs (equal wall-clock, curve
still falling), source `results/run_csv/CAP93K_full162.csv`.
