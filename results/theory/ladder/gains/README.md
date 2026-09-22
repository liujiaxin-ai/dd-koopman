# Ladder weights delivery（`EXPERIMENT_DIRECTIVE_LADDER_WEIGHTS_2026-09-22`）

来源机：`<eval-host>`（hostname `<eval-host>`），目录
`<ladder-out>/gains/`。取件时间：2026-09-22 06:47 UTC，实例为**无卡启动**，
因此本次自检在 CPU 上执行（纯线性代数，与 GPU 无关）。

每个 `.npz` 含 `W`（complex64，`[4, 16]`）与 `W_fp64`（complex128，拟合产物本体）；
`W[p, m]` 把第 `m` 个 Doppler 网格 bin 的系数映射到第 `p` 步的 delay 剖面，
坐标约定 = 训练 warm start 的约定（`delay = ifft_unnormalised(x)`、
`spectrum = fft_unnormalised(delay, time)`）。

## 1. 文件与用途

| 文件 | 拟合数据 | ridge | 形状 | 用在 |
|---|---|---|---|---|
| `W_native.npz` | 打包数据前 **256** 样本（含训练那一次 AWGN） | 1e-6 | `[4,16]` complex | `LSGAINS_K16_{full162,robust486,gen432}.csv`（论文 E1 行） |
| `W_full.npz` | 全部 **24,300** 训练样本 | 1e-6 | `[4,16]` complex | `LSGAINSFULL_K16_{...}.csv`（E1b 对照） |
| `W_full_lambda1e-07.npz` | 24,300 | 1e-7 | `[4,16]` complex | λ 敏感性（`lambda_table.csv`） |
| `W_full_lambda1e-06.npz` | 24,300 | 1e-6 | `[4,16]` complex | λ 敏感性，与 `W_full` 数值相同 |
| `W_full_lambda1e-05.npz` | 24,300 | 1e-5 | `[4,16]` complex | λ 敏感性 |
| `W_periodic.npz` | 解析（无拟合） | 0 | `[4,16]` complex | 归一化交叉验证 `LSGAINSPER_full162.csv`（= DFTGRID K16） |

`fit_summary.json` 为原始拟合记录（未修改）；`manifest.txt` 为 `ls -la` +
`sha256sum` 的原始输出。

## 2. 哈希核对（对任务书 §验收锚点）

| 文件 | 实测 sha256 | 任务书锚点 | 判定 |
|---|---|---|---|
| `W_native.npz` | `40855a5bebbc27bb4f02589e9b503579f803b8621ffff2bab00ac3472dad3c7b` | 同 | **MATCH** |
| `W_full.npz` | `4e394bae7ee31197ced6e5291345bc685ecc8972e38d1f615487b498709633b1` | 同 | **MATCH** |
| `W_full_lambda1e-07.npz` | `7299d7ee3640f7e3765582ff58f2ec0b4490ab7916e1d2b4d3c058ef684ae916` | 同 | **MATCH** |
| `W_full_lambda1e-06.npz` | `0537dd1c82714b1d9bc6462e4370fda4b828f71831263f6ac116a6226dd35827` | 同 | **MATCH** |
| `W_full_lambda1e-05.npz` | `54407f87ba59283dd1162b48b4c81884ea7c691f123794757aef21d73c4f02db` | `54407f87ba59283dd1162b48b4c81884ea7c691f123794757aef21d73a02db` | **锚点仅 62 位（sha256 应 64 位），前 56 位完全一致 ⇒ 判定为任务书转录截断，文件本身一致** |
| `W_periodic.npz` | `b2742e805556215f61e7395852f9cff9c3f8206206d267dc27ad01fac87c6229` | 锚点未列 | 附交 |

本地（本 workspace）重算的哈希与上表逐字节一致。**没有重拟合、没有改 `fit_summary.json`。**

## 3. 实际跑评测的代码版本（任务书 §3）

harness 内（评价时所装）：

| 文件 | sha256 |
|---|---|
| `src/cp/models/baseline/statistical/dftgrid.py` | `394ddbe12af97c7332341dba97b0ec5535d38d4e149df22db2d17437463aaa73` |
| `src/cp/models/baseline/statistical/lsgains.py` | `805681f3c535b3078d00887bd1e1024f7f10e2f342b05ccc39328d8491528096` |
| `src/cp/models/__init__.py`（注册后） | `327d2ec10ea00543a14a8b022e72fc784e89c880669aa49ed354feaa7857dcee` |
| `src/testing/get_models.py`（免 checkpoint 声明后） | `6d09aaa483f61dbf9aceed741f5802b079c0814027dd55e67bf9255e58c57ed5` |

**源码级对照**：本仓库内两份可读副本
`code/ladder_lsgains.py` 与 `<workspace>/dftgrid.py`
的 sha256 与上表前两行**完全相同** ⇒ 交付到你手上的模型代码就是实际跑评测的版本。

完整 harness 改动（2 个新文件 + 4 行注册）见 `results/theory/ladder/ladder_harness_patch.diff`。
两个模型**源码本体**（以及打完补丁的 harness 文件）已放在
`results/theory/ladder/harness_src/`，可供 `ladder_install*.py --source` 直接使用。

## 4. Sanity 重跑（任务书 §4，本轮在打包出的权重上重跑）

命令（harness 根目录）：

```bash
DFTGRID_MODES=16 OMP_NUM_THREADS=8 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  python ladder_sanity.py --out <ladder-out>/sanity_k16_rerun.json
```

完整原始输出：`sanity_rerun_log.txt`；结构化结果：`sanity_k16_rerun.json`。

| 检查 | 本轮实测 | 在案锚点（`sanity_k16.json`） | 判定 |
|---|---|---|---|
| `num_modes` 到达模型 | 16 | 16 | ✓ |
| `topk(16 of 16)` 是否恒等 | true | true | ✓ |
| K=16 预测 = 周期延拓 | **1.345e-06** | 1.72e-06 | ✓ 同量级（float32 舍入，随机初值不同故有差异） |
| 解析 `W_per` vs harness | **1.066e-06** | 1.07e-06 | ✓ 一致到三位 |
| `W_per` 第 0 行 | 0.0625 = 1/16 | 同 | ✓ |
| `passed` | **true** | true | ✓ |

`device = cpu`（本实例无卡启动）；该自检不含任何 GPU 依赖，结论不受影响。

## 5. 复现

```bash
# 拟合（需要 27 子集完整数据 + 约 35 GB 内存；不可与两个训练并行，见 100 GB 上限）
python ladder_lsgains_fit.py --config z_artifacts/config/ours/sweep/tdd_caplong.yaml \
    --out-dir <ladder-out>/gains

# 评测（六个 split 之一）
LSGAINS_W_PATH=<gains>/W_native.npz OMP_NUM_THREADS=8 python eval_ours.py \
    --model LSGAINS --duplex TDD --test-type regular --limit 0 --out /tmp/LSGAINS_full162.csv
```

注意：本机 cgroup 内存上限 **100 GB**、CPU 配额 **20 核**（`nproc` 报 255）。
拟合的重打包阶段若与训练并行会触发 OOM（本次已发生过一次，见
`WRITER_TODO_2026-09-22.md` §6.1）；所有 eval 必须显式设 `OMP_NUM_THREADS`，
否则单 setting 从 2.3 s 劣化到 178 s。
