# DEAP → `eeg_feature_export_v0.1`

在同学侧 SNN 未就绪时，可用 **DEAP** 公开预处理数据跑通「特征 → `train_adapter` → `generate_adapter`」全流程。

## 获取数据

1. 在 [DEAP 官网](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) 注册并下载 **Preprocessed EEG data (Python)**。
2. 解压后应能看到若干 **`s01.dat` … `s32.dat`**（每被试一个 pickle）。

将解压目录记为 `DEAP_PREPROCESSED_DIR`（该目录下直接是 `s01.dat` 等文件，或再嵌套一层，见脚本 `--deap-dir`）。

## 特征与标签（本脚本约定）

- **特征**：每个 trial 对通道维做 **时间均值 + 时间标准差** 拼接 → 维度 **`2 × C`**（预处理版一般为 **C=40** → **80 维**），`float32`。这是**极简基线**，便于接契约，不是论文级 SNN。
- **`emotion_id`（0..3）**：用 **Valence–Arousal 四象限**粗映射到与 EMOPIA 四类训练目标对齐（高/低以量表 midpoint **5** 为界）：

  | 条件 | `emotion_id` | 与 EMOPIA 象限的对应（示意） |
  |------|--------------|------------------------------|
  | V≥5, A≥5 | 0 | 偏「积极高唤醒」→ 对齐 Q1 happy |
  | V<5, A≥5 | 1 | 偏「消极高唤醒」→ 对齐 Q2 angry |
  | V<5, A<5 | 2 | 偏「消极低唤醒」→ 对齐 Q3 sad |
  | V≥5, A<5 | 3 | 偏「积极低唤醒」→ 对齐 Q4 calm |

  该映射仅用于 **Adapter 与现有四类 T5 目标一起训练**，不是 DEAP 论文的官方情绪标签。

## 导出命令

```bash
# 项目根目录；指向含 s01.dat 的目录
python scripts/eeg_public/deap_to_v01.py \
  --deap-dir /path/to/data_preprocessed_python \
  --out res/continuous_features/deap_v01_export.pt

python scripts/eeg_export/validate_pt.py res/continuous_features/deap_v01_export.pt
```

可选：`--subjects s01,s02` 只导部分被试；`--max-trials 10` 试跑。

## 训练

```bash
cd adapter_strategy_v1
python train_adapter.py --eeg-v01 ../res/continuous_features/deap_v01_export.pt
```

`input_dim` 将与导出维数（如 80）一致，已写入 checkpoint。

## 许可

DEAP 有独立使用条款，请仅在遵守其许可的前提下使用与分发导出结果。
