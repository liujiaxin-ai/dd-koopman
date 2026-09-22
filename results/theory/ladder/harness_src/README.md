# Ladder harness sources（追加交付，2026-09-22）

写手侧 workspace 缺的**模型源码本体**。全部取自评测机
`<eval-host>` 的 harness **实际安装文件**（无卡启动），
即跑出 `DFTGRID_K16_*` 与 `LSGAINS_K16_*` 的那一份。

## 1. 新增文件（`ladder_install*.py --source` 的输入）

| 交付路径 | harness 内目标路径 | 说明 |
|---|---|---|
| `src/cp/models/baseline/statistical/dftgrid.py` | 同名 | E0 模型，`class DFTGRIDMODEL`（参数自由；网格宽度由环境变量 `DFTGRID_MODES` 控制） |
| `src/cp/models/baseline/statistical/lsgains.py` | 同名 | E1 模型，`class LSGAINSMODEL`（固定 `[4,16]` 复增益，权重由 `LSGAINS_W_PATH` 指向 `.npz`） |

**逐字节核对**：`dftgrid.py` 与 workspace 的
`<workspace>/dftgrid.py` 相同；
`lsgains.py` 与 workspace 的 `code/ladder_lsgains.py` **完全相同**（哈希见表）。
两份都是可以直接 `--source` 喂给安装脚本、或直接拷进 harness 的文件。

## 2. 修改过的 harness 文件（已含补丁，便于直接比对）

`modified_harness_files/__init__.py`、`modified_harness_files/get_models.py`
是**打完补丁后**的 harness 文件（分别对应 harness 内
`src/cp/models/__init__.py` 与 `src/testing/get_models.py`）。
补丁本身只有 4 处、全文见 `../ladder_harness_patch.diff`：

1. `models/__init__.py`：import `DFTGRIDMODEL`、import `LSGAINSMODEL`；
2. `models/__init__.py`：注册 `DFTGRID_TDD`、`LSGAINS_TDD`；
3. `testing/get_models.py`：`MODELS_NO_CHECKPOINT` 加入 `"DFTGRID"`、`"LSGAINS"`
   （两个基线都无 checkpoint）。

## 3. 哈希

见 `SHA256SUMS.txt`（含与 workspace 副本的对照行）。所有哈希均与
`../gains/manifest.txt` 里“evaluated-code version”一节一致。

## 4. 复现安装（在 harness 根目录）

```bash
python ladder_install.py         --harness <repo> --source <...>/dftgrid.py \
    --diff <out>/ladder_harness_patch.diff
python ladder_install_lsgains.py --harness <repo> --source <...>/lsgains.py \
    --diff <out>/ladder_harness_patch.diff     # 追加到同一个 diff
```
