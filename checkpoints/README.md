# checkpoints/ — 模型权重与指标说明

> ⚠️ **所有权重文件（`*.pt` / `*.pth`，含 100MB 级的 RTMPose 权重）均已加入 `.gitignore`，不提交到仓库。**
> 部署时请按下表准备权重。

---

## 一、当前文件

| 文件 | 大小 | 是否提交 | 说明 |
|------|------|----------|------|
| `best_model.pt` | 6.2 MB | ❌ | **部署模型**：ST-GCN，`in_ch=9`（增强特征），3 分类，2026-08-26 训练 |
| `best_model.metrics.json` | 1.4 KB | ✅ | 上表的评估指标存档（可复现验证） |
| `reported_metrics.json` | 1.4 KB | ✅ | 报告中引用的指标副本 |
| `rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth` | ~100 MB | ❌ | RTMPose-m 姿态模型权重，由 `src/pose_mm.py` 加载 |

### 权重获取方式

```bash
pip install -U openmim

# RTMPose-m 姿态模型（配置 → configs/，权重 → checkpoints/）
mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest ../configs

# RTMDet 人体检测器权重由 mmdet 首次运行时自动下载，无需手动准备
```

---

## 二、部署模型规格（`best_model.pt`）

| 项 | 值 |
|----|-----|
| 模型结构 | ST-GCN（可学习邻接矩阵 + 时间卷积 + 残差） |
| 输入通道 | 9（增强特征：x, y, conf, vx, vy, 相对重心角度, 到重心距离, 弯曲角, 躯干倾角） |
| 滑窗长度 | 30 帧 @ 15fps ≈ 2 秒 |
| 类别数 | 3（`0=正常` / `1=失衡` / `2=摔倒`） |
| 参数量 | ~100 万 |

权重文件中内置元信息：`config`、`num_classes`、`model_type`、`f1`，推理脚本会自动读取并与命令行参数交叉校验。

---

## 三、实测指标（验证集：按视频 8:2 划分，`RandomState(42)`）

| 指标 | 数值 |
|------|------|
| Accuracy | 0.8732 |
| Macro F1 | 0.8197 |
| avg F1（失衡 + 摔倒） | 0.7745 |
| 误报率 fp_rate | 0.3043 |
| score | 0.6224 |

| 类别 | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|-----|-----|-----|
| 0 正常 | 0.9855 | 0.8455 | 0.9102 | 2447 | 36 | 447 |
| 1 失衡 | 0.8660 | 0.9230 | 0.8936 | 743 | 115 | 62 |
| 2 摔倒 | 0.4909 | 0.9859 | 0.6554 | 350 | 363 | 5 |

复现评估：

```bash
cd src
python eval_model.py --checkpoint ../checkpoints/best_model.pt
```

---

## 四、历史版本（`archive/`）

保留历史权重用于回溯对比，**均为已过时版本，请勿用于部署**。

| 文件 | 日期 | 说明 |
|------|------|------|
| `best_model_20260805.pt` | 08-05 | 最早版本，3 通道基础特征 |
| `best_model_20260808.pt` | 08-08 | 多数据集接入后版本 |
| `best_model_20260813.pt` | 08-13 | URFD / Pre-VFall 接入后版本 |
| `best_model_2class_20260826.pt` | 08-26 | 二分类（正常 / 风险）实验版本 |
| `best_stgcn_20260804.pt` | 08-04 | 首个 ST-GCN 训练产物 |

> 归档权重同样被 `.gitignore` 排除（`*.pt`），仅在本机保留。

---

## 五、训练并生成新权重

```bash
cd src

# 单次训练
python train.py --model stgcn --features enhanced --window 30 --epochs 100 --num_classes 3

# 超参搜索（按 score = avgF12 - fp_penalty * fp_rate 选优后自动重训）
python search.py --grid small --epochs 25 --final_epochs 100 --features enhanced
```

产物：
- `checkpoints/best_model.pt` — 最佳权重
- `checkpoints/best_model.metrics.json` — 对应的评估指标

---

## 六、常见问题

| 问题 | 解决 |
|------|------|
| 报错 `in_ch` 不匹配 | 权重是用 `--features basic`（3 通道）训练的，推理需同样用 `--features basic`；或改用增强特征重新训练 |
| 报错找不到 RTMPose 权重 | 按「权重获取方式」下载，并确认文件名与 `src/pose_mm.py` 中 `POSE_CKPT` 一致 |
| 推理结果与报告指标对不上 | 确认加载的是 `best_model.pt`（08-26 版本）而非 `archive/` 中的历史权重 |
| 显存不足 | 降低 `train.py` 的 `--batch_size`；推理端降低 `--infer_width` |
