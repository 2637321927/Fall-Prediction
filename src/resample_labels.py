"""
关键点 + 标签统一重采样工具：把 data/keypoints/ 与 data/labels/ 下所有序列
按时间均匀重采样到统一目标帧率, 使训练集 FPS 一致 (ST-GCN 需要训练/推理一致)。

用法:
    python resample_labels.py --orig_fps 30 --target_fps 15          # 30fps -> 15fps
    python resample_labels.py --orig_fps 30 --target_fps 15 --dry_run # 只预览不写盘

说明:
    - 关键点取区间起点帧(最近帧, 不插值避免伪姿态), 标签取区间内最高等级
    - 两者使用同一时间网格, 保证帧对齐
    - 支持非整数倍 (浮点比例), 例如 25fps -> 15fps
    - 若源混合 25/30fps, 请按源 fps 分两批分别指定 --orig_fps 跑到同一 --target_fps
"""

import numpy as np
from pathlib import Path
import argparse


def resample(labels, orig_fps, target_fps):
    """
    labels: (N,) 原视频帧级标签
    orig_fps: 原视频帧率
    target_fps: 提取帧率
    返回: (M,) 降采样后的标签
    """
    if target_fps >= orig_fps:
        return labels
    ratio = orig_fps / target_fps
    n = len(labels)
    new_n = int(np.ceil(n / ratio))
    new_labels = np.zeros(new_n, dtype=labels.dtype)
    for i in range(new_n):
        start = int(i * ratio)
        end = min(int((i + 1) * ratio), n)
        # 取区间内最高等级
        new_labels[i] = labels[start:end].max()
    return new_labels


def resample_keypoints(kpts, orig_fps, target_fps):
    """
    关键点按时间均匀重采样 (与 resample 使用相同时间网格, 保证帧对齐)

    kpts: (N, 17, 3) 关键点序列
    orig_fps: 原帧率
    target_fps: 目标帧率
    返回: (M, 17, 3) 重采样后的关键点
    """
    if target_fps >= orig_fps:
        return kpts
    ratio = orig_fps / target_fps
    n = len(kpts)
    new_n = int(np.ceil(n / ratio))
    # 每个新帧取对应时间区间的起点帧 (最近帧, 不插值, 避免伪姿态)
    idx = np.array([int(i * ratio) for i in range(new_n)], dtype=np.int64)
    idx = np.clip(idx, 0, n - 1)
    return kpts[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_fps", type=float, default=30, help="源视频/标注的原始帧率")
    parser.add_argument("--target_fps", type=float, default=15, help="统一后的目标帧率")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="数据根目录, 内含 keypoints/ 与 labels/")
    parser.add_argument("--prefix", type=str, default="",
                        help="只处理文件名以该前缀开头的视频(区分大小写); 默认处理全部")
    parser.add_argument("--dry_run", action="store_true", help="只预览不写盘")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    kpt_dir = data_dir / "keypoints"
    label_dir = data_dir / "labels"
    kpt_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    if args.target_fps >= args.orig_fps:
        print(f"[警告] 目标帧率 {args.target_fps} >= 原帧率 {args.orig_fps}, "
              f"不会发生降采样; 若源 fps 不同请按源分别指定 --orig_fps")

    kpt_files = sorted(kpt_dir.glob("*_keypoints.npy"))
    if not kpt_files:
        print(f"未找到关键点文件: {kpt_dir}/*_keypoints.npy")

    for kf in kpt_files:
        name = kf.stem.replace("_keypoints", "")
        if args.prefix and not name.startswith(args.prefix):
            continue
        lf = label_dir / f"{name}_labels.npy"

        k = np.load(kf)
        k_new = resample_keypoints(k, args.orig_fps, args.target_fps)

        has_label = lf.exists()
        if has_label:
            l = np.load(lf)
            l_new = resample(l, args.orig_fps, args.target_fps)
            # 同一时间网格, 帧数应一致; 截断到较短者兜底
            m = min(len(k_new), len(l_new))
            k_new, l_new = k_new[:m], l_new[:m]
        else:
            l_new = None

        msg = f"  {name}: 关键点 {len(k)}→{len(k_new)}帧"
        if has_label:
            msg += f" | 标签 {len(l)}→{len(l_new)}帧"
        else:
            msg += " | 无标签, 跳过"
        print(msg)

        if not args.dry_run:
            np.save(kf, k_new)
            if has_label:
                np.save(lf, l_new)

    if args.dry_run:
        print("\n[预览模式] 未写入任何文件; 去掉 --dry_run 执行实际重采样")

    print("Done")


if __name__ == "__main__":
    main()
