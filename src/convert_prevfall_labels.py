"""
Pre-VFall 标签转换脚本
将官方 CSV 的抽帧标签 → 前向填充到 30fps → *_labels.npy

用法: python convert_prevfall_labels.py
"""
import csv, re, sys, numpy as np
from pathlib import Path
from collections import defaultdict

# 路径配置（全部相对项目根目录解析）
BASE = Path(__file__).resolve().parent.parent   # 项目根目录
CSV_PATH = BASE / "data" / "raw" / "prevfall" / "keypoints_no confidence.csv"
KP_DIR = BASE / "data" / "keypoints"
OUT_DIR = BASE / "data" / "labels"

def video_name(seg_id):
    """'CLFM1_Abnormal' -> 'CLFM1', 'CGFM2' -> 'CGFM2'"""
    return re.sub(r'_(Abnormal|Normal)$', '', seg_id)

def main():
    if not CSV_PATH.exists():
        print(f"CSV 不存在: {CSV_PATH}")
        sys.exit(1)

    # 1. 读 CSV，按视频分组（合并 Normal/Abnormal 段）
    frames_by_video = defaultdict(dict)  # video -> {frame_idx: class}
    with open(CSV_PATH) as f:
        r = csv.DictReader(f)
        for row in r:
            seg_id = row['id'].split('-')[0]       # 'CLFM1_Abnormal' 或 'CGFM2'
            vname = video_name(seg_id)             # 'CLFM1' 或 'CGFM2'
            m = re.search(r'_(\d+)_keypoints', row['id'])
            if not m:
                continue
            fn = int(m.group(1))
            # 重叠帧取较高类别（宁可误报不可漏报）
            cur = frames_by_video[vname].get(fn, -1)
            new_cls = int(row['class'])
            frames_by_video[vname][fn] = max(cur, new_cls)

    print(f"CSV 视频数: {len(frames_by_video)}")
    print(f"关键点目录: {KP_DIR}")

    # 2. 对每个视频生成标签
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []

    for vname in sorted(frames_by_video.keys()):
        kp_file = KP_DIR / f"{vname}_keypoints.npy"
        if not kp_file.exists():
            skipped.append(vname)
            continue

        kp = np.load(kp_file)
        n_frames = len(kp)

        # 3. 前向填充标签
        frame_labels = frames_by_video[vname]
        labels = np.full(n_frames, -1, dtype=np.int64)
        sorted_fns = sorted(frame_labels.keys())

        # 把 CSV 帧号写入对应位置
        for fn in sorted_fns:
            if fn < n_frames:
                labels[fn] = frame_labels[fn]

        # 前向填充：-1 的位置用前面最近的有效标签
        last_valid = 0
        for i in range(n_frames):
            if labels[i] == -1:
                labels[i] = last_valid
            else:
                last_valid = labels[i]

        out_file = OUT_DIR / f"{vname}_labels.npy"
        np.save(out_file, labels)
        done.append((vname, n_frames))

    print(f"\n完成 {len(done)} 个视频:")
    for vname, n in done:
        lbl = np.load(OUT_DIR / f"{vname}_labels.npy")
        from collections import Counter
        cnt = Counter(lbl.tolist())
        print(f"  {vname}: {n}帧, 标签分布={dict(sorted(cnt.items()))}")

    if skipped:
        print(f"\n跳过 {len(skipped)} 个（无对应关键点文件）:")
        for v in skipped:
            print(f"  {v}")

if __name__ == "__main__":
    main()
