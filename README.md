# Fall-Prediction · 基于骨架时序模型的跌倒检测与分级预警

> **Video-based Fall Detection & Graded Early-Warning System**
> 视频 → 人体检测 → 骨架关键点 → 时序深度模型 → 四级预警

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenMMLab](https://img.shields.io/badge/OpenMMLab-mmpose%201.3-1A73E8)](https://github.com/open-mmlab/mmpose)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#许可)

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [技术路线](#技术路线)
- [实测性能](#实测性能)
- [目录结构](#目录结构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [数据集](#数据集)
- [关键设计决策](#关键设计决策)
- [常见问题](#常见问题)
- [文档索引](#文档索引)

---

## 项目简介

本项目面向**居家养老看护场景**，从普通 RGB 视频中实时识别老人跌倒并给出**分级预警**，为照护人员争取干预时间。

系统不依赖穿戴设备，只使用一台普通摄像头（或 RTMP 网络流），完整链路为：

**视频帧 → RTMDet 人体检测 → RTMPose 提取 17 点 COCO 骨架 → 关键点归一化 → 增强特征 → 滑动窗口时序建模（ST-GCN / TCN）→ 状态机四级预警**

输出为三分类时序状态 **正常 / 失衡 / 摔倒**，再由状态机转换为四级预警 **SAFE → LOW_RISK → HIGH_RISK → IMMEDIATE**，可对接 Web / 移动端监控看板。

---

## 核心特性

| 能力 | 说明 |
|------|------|
| 🎯 **骨架时序建模** | 不做图像分类，而是把人体骨架当时间序列建模，抗背景干扰、跨场景泛化更好 |
| 🧠 **双模型路线** | ST-GCN（空间图卷积，理解关节拓扑）与 TCN（因果膨胀卷积，极低延迟）可对比选型 |
| ⚡ **高性能推理** | `fast_infer` 绕开高层后处理包装，手写紧凑后处理，检测 **185ms → 20ms**、姿态 **44ms → 11ms** |
| 🔔 **四级分级预警** | 状态机 + 连续确认机制（避免单帧抖动误报），阈值全参数化可调 |
| 📹 **多视频源** | 支持摄像头、本地视频、RTMP/RTSP 网络流；H.265 流自动用 ffmpeg 转码为 H.264 |
| 🏷️ **完整标注工具链** | 交互式逐帧标注（4 个跌倒关键时间点）、xlsx 批量标注转换、文本标注转换 |
| 🌐 **实时服务能力** | FastAPI 提供 `/api/status`、`/api/history`、`/api/config`、`/ws` 与 MJPEG `/video` 流 |
| 🔁 **训练/推理严格一致** | 归一化、FPS 重采样、特征工程、窗口长度在训练与推理端完全对齐 |

---

## 技术路线

```mermaid
flowchart LR
    A["视频源<br/>摄像头 / RTMP / 视频文件"] --> B["RTMDet<br/>人体检测"]
    B --> C["RTMPose-m<br/>17 点 COCO 骨架"]
    C --> D["归一化<br/>髋居中 + 除身高"]
    D --> E["增强特征<br/>9 通道"]
    E --> F["滑动窗口<br/>30 帧 @ 15fps ≈ 2s"]
    F --> G["时序模型<br/>ST-GCN / TCN"]
    G --> H["状态机<br/>连续确认去抖"]
    H --> I["四级预警<br/>SAFE/LOW/HIGH/IMMEDIATE"]
```

### 输入输出约定

| 环节 | 数据形状 | 说明 |
|------|----------|------|
| 关键点 | `(N, 17, 3)` | N 帧，17 个 COCO 关节，(x, y, confidence)，**不含 FPS 元数据**（FPS 体现为帧数 N） |
| 增强特征 | `(N, 17, 9)` | 原始 x,y,conf + 速度 + 关节角度 + 到重心距离 + 弯曲角 + 躯干倾角 |
| 训练样本 | `(B, T, J, C)` = `(batch, 30, 17, 9)` | 滑窗切片，窗口标签取**尾部 `window//3` 帧**的最高等级 |
| 标签 | `0=正常` / `1=失衡` / `2=摔倒` | 帧级 int64 |

---

## 实测性能

环境：RTX 4070 Laptop · PyTorch 2.0.1+cu118 · mmcv 2.1.0
模型：`checkpoints/best_model.pt`（ST-GCN，`in_ch=9`，3 分类）
验证集：**按视频** 8:2 划分（`RandomState(42)`），56 段视频 / 4054 个滑动窗口

### 总体指标

| 指标 | 数值 |
|------|------|
| Accuracy | **0.8732** (87.32%) |
| Macro F1 | **0.8197** |
| avg F1（失衡 + 摔倒） | **0.7745** |
| 误报率 fp_rate | 0.3043 |
| score = avgF1₂ − 0.5·fp_rate | **0.6224** |

### 分类别明细

| 类别 | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|-----|-----|-----|
| 0 正常 | 0.9855 | 0.8455 | 0.9102 | 2447 | 36 | 447 |
| 1 失衡 | 0.8660 | 0.9230 | 0.8936 | 743 | 115 | 62 |
| 2 摔倒 | 0.4909 | **0.9859** | 0.6554 | 350 | 363 | 5 |

> **结论**：摔倒类召回 **98.59%**（4054 个窗口中仅漏检 5 个），符合监护场景「宁可误报、不可漏报」的目标；正常类精确率 98.55%，日常不会频繁误报。摔倒类精确率偏低（失衡/正常被误判为摔倒）在部署侧由状态机**连续确认**机制压制——IMMEDIATE 需连续 5 次 P₂ > 0.85。

### 推理性能（960×540 测试帧，10 次平均）

| 阶段 | 旧链路 | fast_infer | 提速 |
|------|--------|------------|------|
| 人体检测 | ~185 ms | **20.2 ms** | ≈9× |
| 骨架提取 | ~44 ms | **11.3 ms** | ≈4× |
| 单帧合计 | ~229 ms | **~30 ms** | 可支撑 15fps 实时 |

---

## 目录结构

```text
Fall-Prediction/
├── README.md                    本文件
├── requirements.txt             依赖清单
├── .gitignore / .gitattributes
│
├── src/                         ★ 全部核心代码（详见 src/README.md）
│   ├── pose_mm.py               关键点提取（RTMDet + RTMPose）
│   ├── batch_extract.py         批量关键点提取
│   ├── label_video.py           交互式逐帧标注工具
│   ├── make_labels_from_xlsx.py xlsx 标注 → 帧级标签
│   ├── convert_prevfall_labels.py  Pre-VFall CSV → 标签
│   ├── text_labels_to_npy.py    纯文本标注 → 标签
│   ├── resample_labels.py       FPS 统一重采样（关键点 + 标签同网格）
│   ├── crop_urfd.py             URFD 左右拼接画面裁切
│   ├── dataset.py               滑动窗口切分
│   ├── features.py              增强特征工程（3 → 9 通道）
│   ├── pose_vis.py              骨架绘制
│   ├── stgcn_model.py           ST-GCN 模型
│   ├── tcn_model.py             TCN 模型
│   ├── train.py                 训练入口
│   ├── search.py                超参网格搜索
│   ├── eval_model.py            模型评估
│   ├── state_machine.py         风险分数 → 四级预警
│   ├── infer.py                 离线推理（生成带预警条的结果视频）
│   ├── demo_live.py             实时推理（摄像头 / 视频 / RTMP）
│   ├── detect_videos.py         批量视频检测
│   ├── fast_infer.py            低延迟推理管线
│   ├── low_level_infer.py       低层 API 推理封装
│   ├── server.py                FastAPI 实时服务
│   ├── compare_fast.py          fast_infer 一致性回归
│   ├── compare_keypoints.py     MPII ↔ COCO 关键点一致性校验
│   └── export_data.py           数据导出 + 骨架动画
│
├── configs/                     RTMPose 模型配置
├── docs/                        文档与报告（详见 docs/README.md）
├── annotations/                 人工标注源文件（xlsx + 标注规范）
│
├── data/                        数据根目录（详见 data/README.md）
│   ├── keypoints/               训练用关键点 (N,17,3)
│   ├── labels/                  训练用帧级标签
│   ├── raw/                     原始视频数据集（gitignore）
│   └── variants/                中间版本 / 实验数据
│
├── checkpoints/                 模型权重与指标（详见 checkpoints/README.md）
│   ├── best_model.pt            部署模型（ST-GCN, in_ch=9）
│   ├── best_model.metrics.json  评估指标存档
│   └── archive/                 历史版本权重
│
├── video/                       测试视频
├── result/                      推理结果视频
├── artifacts/                   数据核验产物（骨架动画 / 导出文本）
└── mmpose/                      第三方源码（只读克隆，不纳入版本库）
```

---

## 环境安装

### 1. 基础环境

推荐使用 conda 创建独立环境：

```bash
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

### 2. 安装 PyTorch（CUDA 版）

```bash
# 实测环境为 CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 OpenMMLab 系列

`mmcv` 必须是与 torch / CUDA 匹配的**完整编译版**：

```bash
pip install -U openmim
mim install mmengine "mmcv>=2.1.0" "mmdet>=3.3.0"
pip install "mmpose>=1.3.0"
```

### 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

### 5. 获取 mmpose 源码（可选）

`src/compare_fast.py` 等诊断脚本需要 mmpose 仓库的示例资源：

```bash
git clone https://github.com/open-mmlab/mmpose.git
```

### 6. 验证环境

```bash
python -c "import torch,mmcv,mmdet,mmpose; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmpose.__version__)"
```

预期输出（版本号依实际安装略有差异）：

```text
2.0.1+cu118 True 2.1.0 1.3.0
```

> `True` 表示 GPU 可用；若为 `False`，说明装的是 CPU 版 PyTorch，训练/推理会非常慢。

---

## 快速开始

> ⚠️ **运行位置**：除 `batch_extract.py`、`detect_videos.py`、`make_labels_from_xlsx.py`、`export_data.py`
> 以脚本自身位置解析默认路径外，其余脚本的默认路径（`../data`、`../checkpoints`、`../video`、`../result`）
> 都相对于**当前工作目录**，因此请在 `src/` 目录下执行命令。

### 步骤 0：准备模型权重

```bash
# 下载 RTMPose-m 配置与权重到 configs/ 与 checkpoints/
mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest ../configs
# 权重文件放置到 checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-*.pth
# RTMDet 检测器权重首次运行会自动下载
```

### 步骤 1：提取骨架关键点

```bash
cd src

# 单个视频
python pose_mm.py --video ../video/CGFM1.mp4

# 批量（模型只加载一次），默认递归处理 data/raw 下所有视频
python batch_extract.py --dir ../data/raw/urfd --exts mp4
```

产物：`data/keypoints/<name>_keypoints.npy`，形状 `(N, 17, 3)`。

### 步骤 2：制作帧级标签

三种途径，按数据集标注情况选择：

```bash
# a) 交互式逐帧标注（空格播放/暂停，←→ 逐帧，R/F/I/S 标 4 个时间点，Q 保存）
python label_video.py --video ../video/CGFM1.mp4

# b) xlsx 批量标注 → 标签（先 --dry_run 预览）
python make_labels_from_xlsx.py --dry_run

# c) 公开数据集已有标签转换
python convert_prevfall_labels.py
```

### 步骤 3：统一 FPS（推荐）

混合 25/30fps 源统一到 **15fps**，关键点与标签使用同一时间网格：

```bash
python resample_labels.py --orig_fps 30 --target_fps 15 --dry_run   # 先预览
python resample_labels.py --orig_fps 30 --target_fps 15             # 再执行
```

### 步骤 4：训练

```bash
# ST-GCN + 增强特征 + 3 分类（推荐配置）
python train.py --model stgcn --features enhanced --window 30 --epochs 100 --num_classes 3

# 超参网格搜索（按 score = avgF12 - fp_penalty * fp_rate 自动选优后重训）
python search.py --grid small --epochs 25 --final_epochs 100 --features enhanced
```

产物：`checkpoints/best_model.pt` + `checkpoints/best_model.metrics.json`

### 步骤 5：评估

```bash
python eval_model.py --checkpoint ../checkpoints/best_model.pt
```

### 步骤 6：推理

```bash
# 实时：摄像头
python demo_live.py --checkpoint ../checkpoints/best_model.pt

# 实时：RTMP 网络流（地址必须加引号）
python demo_live.py --checkpoint ../checkpoints/best_model.pt --video "rtmp://..." --profile

# 离线：整段视频 → 带骨架 + 预警条的结果视频
python infer.py --checkpoint ../checkpoints/best_model.pt --video ../video/CGFM1.mp4

# 批量：处理整个 video/ 目录
python detect_videos.py --video_dir ../video --checkpoint ../checkpoints/best_model.pt
```

实时推理的关键参数（**勿随意改动，需与训练一致**）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--infer_fps` | 15 | 推理采样帧率，必须与训练一致 |
| `--window` | 30 | 滑窗长度，必须与训练一致 |
| `--detect_every` | 1 | 每 N 帧做一次人体检测并复用 bbox，调大可提速 |
| `--infer_width` | — | 推理分辨率宽度，2K 流建议 960 / 640 |
| `--profile` | 关 | 打印读帧/检测/姿态/特征/显示分段耗时 |
| `--play_only` | 关 | 只播放不推理，用于单独验证流与转码链路 |

画面效果：视频窗口叠加 17 点骨架，底部为风险条（绿 → 黄 → 橙 → 红）与当前预警等级，按 `Q` 退出。

### 步骤 7：启动 API 服务（可选）

```bash
python server.py --checkpoint ../checkpoints/best_model.pt
```

| 接口 | 说明 |
|------|------|
| `GET /api/status` | 实时状态：`state` / `risk_score` / `alert_level` / `probabilities` / `fps` |
| `GET /api/history` | 历史事件记录 |
| `GET /api/config` | 阈值等配置 |
| `GET /video` | MJPEG 实时视频流（已叠加骨架与风险条） |
| `WS /ws` | WebSocket 实时推送 |

---

## 模块说明

`src/` 内各文件职责的完整说明见 **[`src/README.md`](src/README.md)**，按数据流水线、模型、训练、推理、预警逻辑分组。

完整数据流：

```text
视频 (.mp4)
  │
  ├─[标注阶段]─────────────────────────────────────
  │   pose_mm.py      ──→ *_keypoints.npy  (关键点)
  │   label_video.py  ──→ *_labels.npy     (标签)
  │   resample_labels.py (FPS 对齐)
  │
  ├─[训练阶段]─────────────────────────────────────
  │   train.py
  │     ├─ dataset.py    (滑动窗口)
  │     ├─ features.py   (9 通道增强特征)
  │     ├─ stgcn_model.py / tcn_model.py
  │     └─ → checkpoints/best_model.pt
  │
  └─[推理阶段]─────────────────────────────────────
      infer.py / demo_live.py / server.py
        ├─ pose_mm.py       (提取 + 归一化)
        ├─ features.py      (增强特征)
        ├─ stgcn_model.py / tcn_model.py (加载权重)
        ├─ state_machine.py (风险分数 → 预警等级)
        └─ → result/*_result.mp4 或 实时画面
```

---

## 数据集

训练数据共 **279 段视频**的关键点与帧级标签，三分类（正常 / 失衡 / 摔倒），统一 15fps。

| 数据集 | 场景 | 自带标注 | 本项目处理方式 |
|--------|------|----------|----------------|
| **URFD** | 实验室，跌倒 + 日常活动 | 仅序列级 fall / ADL | 全量提取骨架 + 人工补标阶段 |
| **CAUCAFall** | 居家非受控环境 | fall / no-fall 逐帧 | 全量提取骨架 + 人工补标阶段 |
| **Pre-VFall** | 公开骨架数据集 | 官方 CSV 抽帧标签 | `convert_prevfall_labels.py` 前向填充转换 |

> 传感器类数据集（SisFall / UniMiB / KFall 等）不符合「视频 → 骨架」路线，未纳入主训练；
> KFall 仅参考其跌倒时间点定义。

### 标注规范

跌倒视频需要标注 **4 个关键时间点**（定义见 [`annotations/标注说明.txt`](annotations/标注说明.txt)）：

| 字段 | 含义 |
|------|------|
| `risk_start_frame` | 开始出现明显失衡风险（身体异常前倾/后倾/侧倾、重心明显偏移） |
| `fall_start_frame` | 已进入不可恢复下坠过程 |
| `impact_frame` | 身体首次接触地面/床/椅子等承托物 |
| `stable_fallen_frame` | 倒地后姿态基本稳定，不再明显移动 |

正常视频不填这四个字段，标签全部为 `0`。

数据目录的详细说明见 **[`data/README.md`](data/README.md)**。

---

## 关键设计决策

以下决策均通过**对照实验**得出，实验记录见 [`docs/开发日志.md`](docs/开发日志.md)。

### 1. 按视频划分数据集，而非按帧

同一视频的相邻帧高度相关，若按帧随机切分，训练集与验证集会包含同一视频的邻近帧，等价于**数据泄漏**，验证指标会虚高。因此按视频整体 8:2 划分，并固定 `RandomState(42)` 保证可复现。

### 2. 窗口标签取「尾部 max」而非「整窗 max」

整窗取 max 会把「跌倒前整段正常姿态」提前污染为异常，导致报警提前、假阳性升高。最终采用 `label_tail`：窗口标签只看最后 `window // 3` 帧的最高等级，让标签反映窗口尾部真实状态。

### 3. 增强特征（3 → 9 通道）

| 通道 | 含义 |
|------|------|
| 0–2 | 原始 x, y, confidence |
| 3–4 | 速度 vx, vy |
| 5 | 关节相对重心的角度 |
| 6 | 关节到重心距离 |
| 7 | 关节弯曲角 |
| 8 | 躯干倾角（广播；跌倒时躯干由直立转水平） |

消融实验表明增强特征提升视角鲁棒性与失衡/倒地的区分度，最终 `in_ch=9`。

### 4. 关键点归一化：髋居中 + 除身高

消除相机距离、拍摄角度、被摄者身材与画面分辨率带来的差异，使模型跨场景可用。

### 5. 状态机连续确认，抑制单帧抖动

| 预警等级 | 触发条件 |
|----------|----------|
| `LOW_RISK` | P₁ > 0.60 **连续 3 次** |
| `HIGH_RISK` | P₁ > 0.75 **连续 2 次** |
| `IMMEDIATE` | P₂ > 0.85 **连续 5 次** |

所有阈值与连续次数均已参数化，可按场景调优。

### 6. 指标体系而非单一准确率

数据严重类别不平衡，只报准确率会掩盖最致命的摔倒漏检。因此采用
**「一个总体（Accuracy）+ 一个均衡（Macro F1）+ 两个任务核心（avgF1₂）+ 每类细粒度（P/R/F1 + TP/FP/FN）+ 一个可优化目标（score）」** 的组合。
`score = avgF1₂ − fp_penalty × fp_rate` 用于自动选超参，避免在多指标间人工拍板。

---

## 常见问题

| 问题 | 原因与解决 |
|------|------------|
| `import mmpose / mmdet` 报错 | 当前解释器不在装有 mmcv/mmdet/mmpose 的 conda 环境里。先 `conda activate <环境名>`，或在 VS Code 中选择该环境的解释器 |
| RTMP 流打不开 | ① 地址含 `&` `?` `=` 时必须整体加引号（PowerShell 中 `&` 是命令分隔符）；② H.265 流会自动经 ffmpeg 转 H.264；③ 用 `--play_only` 单独验证解码链路 |
| 实时推理卡在「视频源」后无窗口 | 旧版时间采样 bug（`last_sample_t` 初始化为 0 导致死循环），现版本已修复；若仍卡住，用 `python -u` 启动查看最后输出定位 |
| 检测 2K 大图很慢（每帧 ~1.7s） | 用 `--detect_every 5` 复用 bbox，或用 `--infer_width 960` 缩小检测输入 |
| `fast_infer` 与旧链路关键点差 ~3.6px | 旧链路 `predict` 含 flip_test（TTA）双前向融合，fast 改为单次前向（与训练一致）。960×540 上约 0.3%，阈值 5px 内可忽略 |
| 找不到数据文件 | 确认在 `src/` 目录下执行；或显式传入 `--data_dir` / `--checkpoint` 等参数 |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [`docs/部署与运行说明与系统设计.txt`](docs/部署与运行说明与系统设计.txt) | 从零部署的完整操作说明 + 系统设计与技术要点 |
| [`docs/开发日志.md`](docs/开发日志.md) | 按时间线记录 9 个版本的方案设计、实验测试与问题排查 |
| [`docs/验证设计说明.txt`](docs/验证设计说明.txt) | 为什么选这些指标、实验如何设计、验证是否科学 |
| [`docs/实测数据与功能测试报告.txt`](docs/实测数据与功能测试报告.txt) | 实测精度、推理性能与功能测试结论 |
| [`docs/API.md`](docs/API.md) | HTTP / WebSocket 接口文档（字段、示例、阈值配置） |
| [`docs/萤石开放平台调用证据.txt`](docs/萤石开放平台调用证据.txt) | 萤石开放平台（RTMP 流）调用记录 |
| [`src/README.md`](src/README.md) | 源码模块职责与完整数据流 |
| [`data/README.md`](data/README.md) | 数据目录组织、来源与再生成方式 |
| [`checkpoints/README.md`](checkpoints/README.md) | 权重版本说明与加载方式 |
| [`annotations/标注说明.txt`](annotations/标注说明.txt) | 标注规范与数据组交付要求 |

---

## 开发历程

| 版本 | 时间 | 主要内容 |
|------|------|----------|
| v0.1 | 08-04 ~ 08-05 | 初始框架：RTMPose + ST-GCN/TCN，摄像头 demo |
| v0.2 | 08-06 | 实时推理、多事件标注、批量提取、4 分类支持 |
| v0.3 | 08-07 ~ 08-08 | 多数据集接入、归一化对齐、增强特征、URFD 预处理 |
| v0.4 | 08-11 ~ 08-13 | URFD / Pre-VFall 数据、关键点一致性校验、离线推理 |
| v0.5 | 08-19 ~ 08-24 | Web API 服务：HTTP + WebSocket + MJPEG 视频流 |
| v0.6 | 08-24 ~ 08-25 | 推理性能优化：`fast_infer` 低延迟管线 |
| v0.7 | 08-25 | 推理服务迁移接入 CareWatch 后端（落库 / 告警 / 推送） |
| v0.8 | 08-26 | 数据 FPS 统一、状态机调参、最终模型训练与评估 |
| v0.9 | 08-27 ~ 08-28 | 批量检测、前后端联调收尾 |

---

## 许可

本项目为**学术研究 / 课程设计**用途。使用的公开数据集（URFD、CAUCAFall、Pre-VFall）版权归各自作者所有，
使用时请遵循其原始许可与引用要求。第三方组件（mmpose / mmdet / mmcv / mmengine）遵循 Apache-2.0 许可。

---

<p align="center"><i>如果这个项目对你有帮助，欢迎 Star ⭐</i></p>
