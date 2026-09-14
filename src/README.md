# src/ 模块职责说明

> **运行位置**：除 `batch_extract.py`、`detect_videos.py`、`make_labels_from_xlsx.py`、`export_data.py`
> 以脚本自身位置解析默认路径外，其余脚本的默认路径（`../data`、`../checkpoints`、`../video`、`../result`）
> 均相对于**当前工作目录**，因此请在 `src/` 目录下执行命令：
> ```bash
> cd src
> python train.py --model stgcn --features enhanced
> ```
> 若需从其他目录执行，请显式传入 `--data_dir` / `--checkpoint` 等参数。

## 一、数据流水线（提取 → 标注 → 处理）

| 文件 | 职责 |
|------|------|
| `pose_mm.py` | **关键点提取**。用 RTMDet 检测人体 + RTMPose 提取 17 个 COCO 关键点，归一化（髋居中 + 除身高），保存为 `*_keypoints.npy`，可选生成骨架标注视频 |
| `batch_extract.py` | **批量提取**。遍历目录下所有 .mp4，调用 `pose_mm` 逐个提取，模型只加载一次 |
| `label_video.py` | **交互式标注**。播放视频，键盘打标签（正常/失衡/倒地），输出 `*_labels.npy` |
| `crop_urfd.py` | **URFD 预处理**。将 URFD 左右拼接画面裁剪为仅保留左半（深度图） |
| `resample_labels.py` | **标签重采样**。当标签帧数与关键点帧数不一致时，重采样对齐 |

## 二、数据集

| 文件 | 职责 |
|------|------|
| `dataset.py` | **滑动窗口切分**。把 (N, 17, 3) 关键点 + (N,) 标签切为固定窗口（如 30 帧），窗口标签取窗口内最大值，输出 (window, 17, channels) 样本 |

## 三、特征工程

| 文件 | 职责 |
|------|------|
| `features.py` | **增强特征**。从基础 (x,y,conf) 衍生出速度、角度、重心距离等，输出 (N, 17, 7) 增强特征 |
| `pose_vis.py` | **骨架绘制**。在画面上绘制 COCO 17 点骨架连线，用于可视化 |

## 四、模型

| 文件 | 职责 |
|------|------|
| `stgcn_model.py` | **ST-GCN 模型**。空间图卷积（邻接矩阵聚合相邻关节）+ 时间卷积，理解骨架结构 |
| `tcn_model.py` | **TCN 模型**。因果时间卷积，把关键点展平后沿时间轴卷，结构简单但无空间概念 |

## 五、训练

| 文件 | 职责 |
|------|------|
| `train.py` | **训练脚本**。加载关键点+标签 → 增强特征 → 切窗口 → 训练 ST-GCN/TCN → 保存最佳模型到 `checkpoints/` |

## 六、推理

| 文件 | 职责 |
|------|------|
| `infer.py` | **离线推理**。先全视频提取关键点 → 滑动窗口推理 → 生成带骨架+预警条的 `result/*_result.mp4` |
| `demo_live.py` | **实时推理**。摄像头/视频流，边提取边推理边显示，滑动窗口缓冲区，FPS 显示 |
| `demo_webcam.py` | **早期摄像头 demo**（废弃，功能已被 `demo_live.py` 覆盖） |

## 七、预警逻辑

| 文件 | 职责 |
|------|------|
| `state_machine.py` | **状态机**。将模型输出的风险分数转换为四级预警：SAFE → LOW_RISK → HIGH_RISK → IMMEDIATE |

---

## 完整数据流

```
视频 (.mp4)
  │
  ├─[标注阶段]─────────────────────────────────────
  │   pose_mm.py ──→ *_keypoints.npy (关键点)
  │   label_video.py ──→ *_labels.npy (标签)
  │   resample_labels.py (对齐帧数)
  │
  ├─[训练阶段]─────────────────────────────────────
  │   train.py
  │     ├─ dataset.py (切窗口)
  │     ├─ features.py (增强特征)
  │     ├─ stgcn_model.py / tcn_model.py (模型)
  │     └─ → ../checkpoints/best_model.pt
  │
  └─[推理阶段]─────────────────────────────────────
      infer.py / demo_live.py
        ├─ pose_mm.py (提取关键点+归一化)
        ├─ features.py (增强特征)
        ├─ stgcn_model.py / tcn_model.py (加载权重)
        ├─ state_machine.py (风险分数→预警等级)
        └─ → result/*_result.mp4 或 实时画面
```

---

## 完整文件清单

| 文件 | 分类 | 职责 |
|------|------|------|
| `pose_mm.py` | 数据 | RTMDet + RTMPose 提取 17 点骨架，归一化后存 npy |
| `batch_extract.py` | 数据 | 批量提取（模型只加载一次），按源目录镜像输出 |
| `label_video.py` | 数据 | 交互式逐帧标注（4 个跌倒关键时间点） |
| `make_labels_from_xlsx.py` | 数据 | xlsx 分段标注 → 帧级标签，含冲突检测 |
| `convert_prevfall_labels.py` | 数据 | Pre-VFall 官方 CSV → 前向填充标签 |
| `text_labels_to_npy.py` | 数据 | 纯文本标注（稀疏 / 逐帧）→ 标签 |
| `resample_labels.py` | 数据 | 混合 FPS 统一重采样，关键点与标签同时间网格 |
| `crop_urfd.py` | 数据 | URFD 左右拼接画面裁切（保留右半 RGB） |
| `export_data.py` | 数据 | 导出关键点文本 + 骨架动画到 `artifacts/` |
| `compare_keypoints.py` | 数据 | MPII ↔ COCO 关键点坐标一致性校验 |
| `dataset.py` | 数据集 | 滑动窗口切分，窗口标签取尾部 `window//3` 帧最高等级 |
| `features.py` | 特征 | 增强特征工程（3 → 9 通道） |
| `pose_vis.py` | 特征 | COCO 17 点骨架绘制 |
| `stgcn_model.py` | 模型 | ST-GCN：可学习邻接矩阵空间图卷积 + 时间卷积 + 残差 |
| `tcn_model.py` | 模型 | TCN：因果膨胀卷积，参数量小、延迟低 |
| `train.py` | 训练 | 训练入口，保存最佳权重与指标 |
| `search.py` | 训练 | 超参网格搜索，按 `score` 选优后重训 |
| `eval_model.py` | 训练 | 加载权重评估，输出 P/R/F1、acc、macro_f1、score |
| `state_machine.py` | 预警 | 风险分数 → 四级预警，连续确认去抖 |
| `infer.py` | 推理 | 离线推理，生成带骨架 + 预警条的结果视频 |
| `demo_live.py` | 推理 | 实时推理（摄像头 / 视频 / RTMP），支持 H.265 自动转码 |
| `demo_webcam.py` | 推理 | 早期摄像头 demo（已废弃，功能被 `demo_live.py` 覆盖） |
| `detect_videos.py` | 推理 | 批量视频检测并汇总结果 |
| `fast_infer.py` | 推理 | 低延迟管线：手写紧凑后处理替代高层包装 |
| `low_level_infer.py` | 推理 | mmdet / mmpose 低层 API 推理封装（对照基线） |
| `compare_fast.py` | 验证 | `fast_infer` vs 低层 API 一致性回归（bbox < 3px、kpts < 5px 判 PASS） |
| `server.py` | 服务 | FastAPI：`/api/status`、`/api/history`、`/api/config`、`/ws`、`/video` |
| `test_ws.py` | 服务 | WebSocket 接口连通性测试 |

---

## 依赖关系

```text
train.py      → dataset.py, features.py, stgcn_model.py, tcn_model.py
eval_model.py → train.py, dataset.py, stgcn_model.py, tcn_model.py
search.py     → train.py, dataset.py, stgcn_model.py, tcn_model.py
infer.py      → stgcn_model.py, tcn_model.py, state_machine.py
batch_extract.py → pose_mm.py
compare_fast.py  → pose_mm.py, low_level_infer.py, fast_infer.py
```

所有 `import` 均为**同目录扁平导入**，因此请勿把脚本拆入子包（会破坏导入关系）。
