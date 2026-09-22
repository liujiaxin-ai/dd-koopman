# 精确性阶梯：零训练评测交付（EXPERIMENT_DIRECTIVE_LADDER_2026-09-21）

日期：2026-09-22（北京）。状态：**任务 A、任务 B 全部完成并落盘**，无训练（只有闭合解岭 LS）。

## 0. 一句话

把"满网格精确传递"（E0）和"同类参数化但解除 exactness、改由数据拟合"（E1）都跑出来了：
**E0 是最差的一档**。三 split 方向一致，阶梯闭合。

## 1. 阶梯（cell 内 6 SNR 先均值再 mean，与现有 ladder 同协议）

| split | settings | E0: DFTGRID K=16（满网格精确 P₀） | E1: LS 拟合增益（native 配方） | E1b: LS 拟合增益（全 24,300 拟合） | 参照: K=1 自适应 top-1 | 参照: E3 训练模型 |
|---|---|---|---|---|---|---|
| regular | 162 | **3.2883** | 1.2504 | 0.8173 | 1.3771 | 0.1402（reported，三种子） |
| robustness | 486 | **2.7558** | 1.1366 | 0.7625 | 1.2465 | 0.1772（seed42） |
| generalization-432 | 432 | **2.7518** | 1.4767 | 1.0545 | 1.2790 | 0.2685（seed42） |

读法（三 split 一致，无一例外）：

1. **满网格精确传递是最差的一档**：3.29 / 2.76 / 2.75，比 K=1 的自适应 top-1 还差 1.9–2.4。
   即"精确"（不截断、周期延拓、Proposition 1 的 P₀ 本尊）**不是**预测好的原因；
   截断在这里起的是平滑/正则作用。
2. **解除 exactness、用同一参数化拟合同一个 [4×16] 增益矩阵**，立刻从 3.29 掉到 1.25（native 配方）
   或 0.82（全量拟合）——即自由度的价值在读出增益上，与理论一致。
3. 距离训练模型（0.14）仍远：E1 只用了固定的、无条件化的 64 个复数增益，
   没有 denoiser、没有 per-sample 条件化、没有门控。

## 2. 任务 A：DFTGRID K=16（E0）

* 配置：`DFTGRID_MODES=16`，与 K1/K2/K4/K8 **同一条 harness 路径**（同一个 `eval_ours.py` →
  `test_unit`），三 split 全量。
* 输出：`results/run_csv/DFTGRID_K16_full162.csv`（162 行）、`_robust486.csv`（486 行）、
  `_gen432.csv`（432 行）；schema 与 `DFTGRID_K*` 逐列一致。
* 完整 K 阶梯（regular，cell-mean NMSE）：
  K=1 **1.3771** → K=2 **1.7825** → K=4 **2.5235** → K=8 **3.0204** → K=16 **3.2883**，
  单调递增，K=16 落在任务书预期区间 3.0–3.5 内。**不是 bug，是预期结果。**
* sanity check（原始输出 `results/theory/ladder/sanity_k16.json`）：

  | 检查 | 值 |
  |---|---|
  | `num_modes` 到达模型 | 16（无静默默认） |
  | topk(16 of 16) 是否恒等选择 | true |
  | K=16 预测 = 周期延拓 `x[t mod 16]` | 最大绝对误差 **1.72e-6**（float32 量级） |
  | 解析 `W_per[p,m] = z_m^p / N` 与 harness 一致 | 最大绝对误差 **1.07e-6**（归一化验证通过） |
  | `W_per` 第一行 | 0.0625 = 1/16（实部），解析正确 |

## 3. 任务 B：LS 拟合网格增益（E1）

* 拟合对象 = 模型自身 warm start 拟合的那个量：`W ∈ C^{4×16}`，tap 共享、跨天线共享、
  逐 horizon、**无条件化**；一次闭合解岭 LS（complex128），无梯度训练。
* 坐标（与训练 warm start 完全一致）：`delay = ifft_unnormalised(x)[..., :300]`、
  `spectrum = fft_unnormalised(delay, time)`、`target = ifft_unnormalised(future)`；
  `W` 由 `gram = DᴴD + (ridge·scale + 1e-12)I` 求解。
* native 配方（= 报告模型实际用的）：`pl.seed_everything(42)` 后重打包 27 子集，
  取 `H_hist[:256]` / `H_pred[:256]`（**带训练时那一次 AWGN 抽样**），ridge = 1e-6。
* 三个拟合的产物与 SHA256 在 `results/theory/ladder/fit_summary.json`：
  `W_native.npz`（256 样本）、`W_full.npz`（24,300 样本）、`W_full_lambda{1e-7,1e-6,1e-5}.npz`。
  拟合 dtype = complex128（存储同时给 fp64 与 fp32）。
* 输出：`results/run_csv/LSGAINS_K16_{full162,robust486,gen432}.csv`（native，论文用）与
  `LSGAINSFULL_K16_{...}.csv`（全量拟合变体），schema 同 `DFTGRID_K*`。
* **交叉验证（任务书要求）**：解析周期延拓增益 `W_per[p,m] = z_m^p/N` 走同一条 LSGAINS eval 管线：
  `LSGAINSPER_full162.csv` 的 cell-mean = **3.2883**，与 harness 的 DFTGRID K=16 **完全相同**；
  逐设定 `nmse_mean` 最大绝对差 **5.0e-06**、平均 **2.5e-07**（float32 舍入）。
  ⇒ 归一化、坐标约定、eval 管线三者都被这一条交叉检查钉住，**没有 silently 选一个数**。

## 4. λ 敏感性（验证集，只进交付摘要、不进论文）

来源 `results/theory/ladder/lambda_table.csv`（验证折 = 每格最后 100 样本，归一化数据，
相对比较用）：

| 拟合 | ridge | 验证集 cell-mean NMSE |
|---|---|---|
| 解析 P₀（periodic） | 0 | 1.7380 |
| native（256 样本） | 1e-6 | 0.9748 |
| full（24,300） | 1e-7 | 0.7169 |
| full（24,300） | 1e-6 | 0.7169 |
| full（24,300） | 1e-5 | 0.7169 |

λ 在 1e-7…1e-5 内**完全不影响**结果（Gram 在这个尺度下条件数很好），
所以论文里的 E1 数字对 λ 不敏感，不需要讨论 ridge。

## 5. harness 改动（任务书 §5.6 要求全文记录）

`DFTGRID` 与 `LSGAINS` 原先都不在跑数据的那台 harness 里，需要端口。完整 unified diff：
`results/theory/ladder/ladder_harness_patch.diff`（3.2 KB）。共两类改动：

1. 新增两个文件（逐字拷贝，未修改）：
   `src/cp/models/baseline/statistical/dftgrid.py`、`.../lsgains.py`；
2. 注册与免 checkpoint 声明：
   `src/cp/models/__init__.py` 加两行 import + 两行 registry（`DFTGRID_TDD`、`LSGAINS_TDD`）；
   `src/testing/get_models.py` 的 `MODELS_NO_CHECKPOINT` 加 `"DFTGRID"`、`"LSGAINS"`。

**没有改动任何训练/评测语义**，`test_unit`、`repro_official.py`、`train_sweep.py` 均未动。

## 6. 运行环境上的一个硬发现（会影响后续所有 eval）

这台实例的 cgroup 配额是 **20 CPU**，但 `nproc` 报 255。eval 进程若不显式设
`OMP_NUM_THREADS`，torch 会开 255 个线程，噪声循环与 collate 里的小张量算子从
~0.4 ms 劣化到 ~340 ms：

* 单个 setting：**178.3 s**（未设）→ **2.3 s**（`OMP_NUM_THREADS=8`），差 **77×**；
* 冷读/热读、batch 1/32/128 都不影响这 178 s（已实测排除 I/O 与 batch 两个假设）；
* 结论：本机所有 eval 命令都必须带 `OMP_NUM_THREADS`。V7 的评估脚本
  `results/theory/gain_cond_ablation/eval_v7.py` 已同步修正。

## 7. 运行时间与复现

| 阶段 | 墙钟 |
|---|---|
| 任务 A：三 split 并行（162+486+432 设置） | 17:31→17:41 ≈ **10 min** |
| 任务 B 拟合：重打包 245 s + native 2 s + full 169 s + λ×3 ≈ 12 min | ≈ **12 min** |
| 任务 B 评测：7 路并行（3+3 split + 1 交叉验证） | 17:42→17:54 ≈ **13 min** |
| λ 表 | ~1 min |

脚本（均在 `code/`，文件头带复现命令）：

| 脚本 | 作用 |
|---|---|
| `code/ladder_install.py` / `code/ladder_install_lsgains.py` | 端口两个基线进 harness 并产出 diff |
| `code/ladder_sanity.py` | E0 的周期延拓/归一化 sanity |
| `code/ladder_lsgains_fit.py` | 重打包 + 三个岭 LS 拟合 |
| `code/ladder_lsgains.py` | LSGAINS harness 模型（读 `LSGAINS_W_PATH`） |
| `code/ladder_run_dftgrid.sh` / `code/ladder_run_lsgains.sh` | 两批评测的运行器（含 OMP 设置） |
| `code/analysis/classical_ladder_extend.py` | 再生 `results/analysis/classical_ladder.csv`（含 K=1 改名） |
| `code/ladder_lambda_table.py` | λ / native / full 的验证集对照 |

一行复现（在 harness 根目录，`DFTGRID_MODES=16 OMP_NUM_THREADS=6`）：

```bash
OMP_NUM_THREADS=6 DFTGRID_MODES=16 python eval_ours.py --model DFTGRID --duplex TDD \
  --test-type regular --limit 0 --out /tmp/DFTGRID_K16_full162.csv
```

## 8. 边界与未做项

* K=2/K=4/K=8 只有 regular 的参照行（任务书只要求 K=16 补齐三 split），未补 robust/gen。
* 交叉验证只在 regular 的 162 设定上做了（其余 split 未重复）。
* λ 表用的是**验证折**而非 test split，只作相对比较，未进论文。
* `LSGAINSFULL`（全量拟合）是任务书要求的第二份对照，**已交付但论文章节自定是否引用**。
* 未改动 `paper/`、`figures/`、audit 脚本、`NARRATIVE_REPORT.md`，也未触碰 THEORY 票的交付物。
