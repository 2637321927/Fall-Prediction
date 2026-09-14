"""
纯文本标注 → *_labels.npy 转换工具（不依赖图形标注工具）

支持两种纯文本标注格式（自动识别）：

  格式 A（推荐，只标变化帧，前向填充）:
      # 注释行以 # 开头
      帧号 标签       例如:  0 0
      338 1
      # 标签含义: 0=正常, 1=跌倒(异常), 2=已倒地(可选3分类)
      # 从该帧起到下一个标注帧之间, 标签沿用该值（前向填充）

  格式 B（全量逐帧, 每帧一行一个标签）:
      0
      0
      1
      1
      ...

用法:
  python text_labels_to_npy.py --txt CGFM2_labels.txt
  python text_labels_to_npy.py --txt CGFM2_labels.txt --out my_labels.npy

输出:
  data/labels/<name>_labels.npy   (对齐 <name>_keypoints.npy 的帧数, int64)
"""
import argparse
import numpy as np
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent          # 项目根目录
KP_DIR = BASE / "data" / "keypoints"
OUT_DIR = BASE / "data" / "labels"


def parse_txt(txt_path):
    """解析标注文件 -> (mode, data)
    mode='sparse': data={frame_idx: label}  变化点标注
    mode='dense':  data=[label, label, ...]  逐帧标注
    """
    sparse = {}
    dense = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 去掉行尾注释
            line = re.sub(r"#.*", "", line).strip()
            if not line:
                continue
            parts = re.split(r"[\s,，]+", line)
            if len(parts) == 2:
                # 格式 A: "帧号 标签"
                fr, lab = int(parts[0]), int(parts[1])
                if fr < 0:
                    raise ValueError(f"帧号不能为负: {line}")
                sparse[fr] = lab
            elif len(parts) == 1:
                # 格式 B: 逐帧一个标签
                dense.append(int(parts[0]))
            else:
                raise ValueError(f"无法解析的行: {line!r}")
    if sparse and dense:
        raise ValueError("同一文件不能混用格式 A 和格式 B")
    if sparse:
        return "sparse", sparse
    if dense:
        return "dense", dense
    raise ValueError(f"标注文件为空: {txt_path}")


def build_labels(name, mode, data):
    """生成对齐关键点帧数的标签数组 (N,) int64"""
    kp_file = KP_DIR / f"{name}_keypoints.npy"
    if not kp_file.exists():
        # 找不到关键点就按标注文件自身长度
        print(f"  [警告] 未找到 {kp_file}, 标签长度按标注文件本身确定")
        n = len(data) if mode == "dense" else (max(data) + 1)
    else:
        n = len(np.load(kp_file))

    labels = np.zeros(n, dtype=np.int64)

    if mode == "sparse":
        # 写入变化点
        for fr, lab in data.items():
            if fr >= n:
                print(f"  [警告] 帧号 {fr} 超出 {name} 的 {n} 帧, 已忽略")
                continue
            labels[fr] = lab
        # 前向填充: 未标注位置沿用前面最近的有效标签
        last = 0
        for i in range(n):
            if labels[i] == 0:
                labels[i] = last
            else:
                last = labels[i]
    else:
        # dense: 长度可能不齐, 裁齐/补尾部
        if len(data) >= n:
            labels = np.asarray(data[:n], dtype=np.int64)
        else:
            arr = np.asarray(data, dtype=np.int64)
            pad = np.full(n - len(arr), arr[-1] if len(arr) else 0)
            labels = np.concatenate([arr, pad])

    return labels


def main():
    ap = argparse.ArgumentParser(description="纯文本标注 → npy")
    ap.add_argument("--txt", required=True, help="标注文本文件路径")
    ap.add_argument("--name", default=None,
                    help="视频名, 用于匹配 data/keypoints/<name>_keypoints.npy"
                         "(默认取标注文件名, 如 CGFM2_labels.txt -> CGFM2)")
    ap.add_argument("--out", default=None, help="输出 npy 路径(默认 data/labels/<name>_labels.npy)")
    args = ap.parse_args()

    txt = Path(args.txt)
    if not txt.exists():
        print(f"标注文件不存在: {txt}")
        return

    # 视频名: 优先 --name, 否则取标注文件名: CGFM2_labels.txt -> CGFM2
    name = args.name if args.name else txt.stem.replace("_labels", "")

    mode, data = parse_txt(txt)
    labels = build_labels(name, mode, data)

    out_dir = OUT_DIR if args.out is None else Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{name}_labels.npy" if args.out is None else Path(args.out)

    np.save(out_file, labels)
    from collections import Counter
    cnt = Counter(labels.tolist())
    print(f"\n视频: {name}  格式: {mode}  帧数: {len(labels)}")
    print(f"标签分布: {dict(sorted(cnt.items()))}")
    print(f"已保存: {out_file}")


if __name__ == "__main__":
    main()
