"""
数据集处理: 滑动窗口切分 + 标签生成
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class FallDetectionDataset(Dataset):
    """
    跌倒检测数据集

    输入:
        keypoints: (N_frames, 17, 3) 或 [(N1,17,3), (N2,17,3), ...] 多视频列表
        labels: (N_frames,) 或 [(N1,), (N2,), ...] 对应标签
        window_size: 窗口帧数 (默认16)
        stride: 滑动步长 (默认4)

    输出每个样本:
        data: (window_size, 17, channels)
        label: int  ← 取窗口末尾 label_tail 帧的最高等级(方案2), 而非整窗 max
    属性:
        video_ids: 每个样本所属视频的索引（用于按视频切分验证集）
    """

    def __init__(
        self,
        keypoints,
        labels,
        window_size: int = 16,
        stride: int = 4,
        label_tail: Optional[int] = None
    ):
        """
        label_tail: 窗口标签只看"最后 N 帧"的最高等级, 而非整窗 max。
                    默认 None = max(1, window_size // 3)。
                    作用: 避免"跌倒前整段正常姿态被窗口 max 提前污染",
                    让标签反映窗口尾部(最近时刻)的真实状态, 报警更即时、假阳性更低。
        """
        self.window_size = window_size
        self.stride = stride
        if label_tail is None:
            label_tail = max(1, window_size // 3)
        self.label_tail = label_tail
        self.samples = []
        self.video_ids = []

        # 支持单个数组或视频列表
        if isinstance(keypoints, list):
            kpts_list, labels_list = keypoints, labels
        else:
            kpts_list, labels_list = [keypoints], [labels]

        for vid, (kpts, lbl) in enumerate(zip(kpts_list, labels_list)):
            n_frames = len(kpts)
            for start in range(0, n_frames - window_size + 1, stride):
                end = start + window_size
                window_kpts = kpts[start:end]      # (window_size, 17, 3)
                window_labels = lbl[start:end]
                # 方案2: 只看窗口尾部 label_tail 帧的最高等级,
                # 窗口尾部进入异常区才标异常, 减少提前污染
                label = int(window_labels[-self.label_tail:].max())
                self.samples.append((window_kpts, label))
                self.video_ids.append(vid)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kpts, label = self.samples[idx]
        return (
            torch.FloatTensor(kpts),   # (window_size, 17, channels)
            torch.LongTensor([label])[0]
        )


def create_sliding_windows(
    keypoints: np.ndarray,
    window_size: int = 16,
    stride: int = 4
) -> np.ndarray:
    """
    纯 numpy 滑动窗口切分（不依赖 Dataset）

    返回:
        windows: (num_windows, window_size, 17, 3)
    """
    n_frames = keypoints.shape[0]
    windows = []

    for start in range(0, n_frames - window_size + 1, stride):
        windows.append(keypoints[start:start + window_size])

    return np.array(windows, dtype=np.float32)


def generate_labels_from_timeline(
    num_frames: int,
    fps: float,
    t0_frame: int,
    t1_frame: int,
    safe_label: int = 0,
    risk_label: int = 1
) -> np.ndarray:
    """
    根据标注时间点自动生成帧级标签

    参数:
        num_frames: 视频总帧数
        fps: 视频帧率
        t0_frame: 开始失衡的帧索引
        t1_frame: 撞地的帧索引

    规则:
        t0 之前 → safe
        t0 ~ t1 及之后 → risk
    """
    labels = np.full(num_frames, safe_label, dtype=np.int64)
    labels[t0_frame:] = risk_label
    return labels


class MultiVideoDataset(Dataset):
    """
    多视频合并数据集
    """

    def __init__(
        self,
        video_paths: List[str],
        label_paths: List[str],
        window_size: int = 16,
        stride: int = 4,
        label_tail: Optional[int] = None
    ):
        self.window_size = window_size
        self.stride = stride
        if label_tail is None:
            label_tail = max(1, window_size // 3)
        self.label_tail = label_tail
        self.samples = []

        for video_path, label_path in zip(video_paths, label_paths):
            kpts = np.load(video_path)              # (N, 17, 3)
            labels = np.load(label_path)            # (N,)

            n_frames = len(kpts)
            for start in range(0, n_frames - window_size + 1, stride):
                end = start + window_size
                window_kpts = kpts[start:end]
                window_labels = labels[start:end]
                # 方案2: 只看窗口尾部 label_tail 帧的最高等级
                label = int(window_labels[-self.label_tail:].max())
                self.samples.append((window_kpts, label))

        print(f"多视频数据集加载完成: {len(self.samples)} 个窗口样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kpts, label = self.samples[idx]
        return (
            torch.FloatTensor(kpts),
            torch.LongTensor([label])[0]
        )


if __name__ == "__main__":
    # 测试
    dummy_kpts = np.random.randn(100, 17, 3).astype(np.float32)
    dummy_labels = np.array([0]*50 + [1]*50)

    ds = FallDetectionDataset(dummy_kpts, dummy_labels, window_size=16, stride=4)
    print(f"窗口样本数: {len(ds)}")
    print(f"样本形状: {ds[0][0].shape}")
    print(f"标签: {ds[0][1]}")

    loader = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"\n批次形状: {batch[0].shape}")
    print(f"标签形状: {batch[1].shape}")
