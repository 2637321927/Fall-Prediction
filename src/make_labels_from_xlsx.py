"""
从标注 xlsx 生成帧级标签 labels.npy (匹配对应关键点长度)

标注格式: 第一列=视频名, 之后每 2 列一组 = [起始帧, 状态]
  状态: 0=正常, 1=不正常, 2=摔倒; 起始帧为 1-based; 最后一个状态持续到视频结束

用法: python make_labels_from_xlsx.py [--xlsx ...] [--kpt_dir ...] [--label_dir ...]
"""
import numpy as np, openpyxl, argparse
from pathlib import Path

# 项目根目录（本脚本位于 <项目>/src/）
BASE = Path(__file__).resolve().parent.parent

DEFAULT_ROOT = BASE / "annotations"
DEFAULT_XLSX = ["标注.xlsx", "h标注.xlsx", "标注o.xlsx", "L标注2.xlsx"]


def parse_row(r):
    """解析一行: (名称, [(起始帧, 状态), ...])"""
    name = str(r[0]).strip() if r[0] is not None else ""
    segs = []
    i = 1
    while i + 1 < len(r):
        s, st = r[i], r[i + 1]
        if s is None or str(s).strip() == "":
            break
        segs.append((int(float(str(s))), int(float(str(st)))))
        i += 2
    return name, segs


def build_labels(n_frames, segs):
    """由 (起始帧[1-based], 状态) 段生成帧级标签"""
    labels = np.zeros(n_frames, dtype=np.int64)
    for j, (start, state) in enumerate(segs):
        start_idx = start - 1
        end_idx = segs[j + 1][0] - 1 if j + 1 < len(segs) else n_frames
        start_idx = max(0, min(start_idx, n_frames))
        end_idx = max(start_idx, min(end_idx, n_frames))
        labels[start_idx:end_idx] = state
    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", nargs="+", default=DEFAULT_XLSX, help="标注文件(相对root或绝对)")
    p.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="标注文件所在根目录")
    p.add_argument("--kpt_dir", type=str, default=str(BASE / "data" / "keypoints"))
    p.add_argument("--label_dir", type=str, default=str(BASE / "data" / "labels"))
    p.add_argument("--dry_run", action="store_true", help="只预览不写盘")
    args = p.parse_args()

    root = Path(args.root)
    kpt_dir = Path(args.kpt_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # 关键点文件名: 小写名 → 实际名(处理大小写不匹配)
    kpt_names = {}
    for f in kpt_dir.glob("*_keypoints.npy"):
        n = f.stem.replace("_keypoints", "")
        kpt_names[n.lower()] = n

    created, skipped, conflicts = [], [], []
    for fn in args.xlsx:
        fpath = Path(fn) if Path(fn).exists() else root / fn
        wb = openpyxl.load_workbook(fpath, read_only=True, data_only=True)
        for ws in wb.worksheets:
            for r in ws.iter_rows(values_only=True):
                if not r or r[0] is None:
                    continue
                name, segs = parse_row(r)
                if not name:
                    continue
                key = name.lower()
                if key not in kpt_names:
                    skipped.append(name)
                    continue
                real = kpt_names[key]
                n_frames = len(np.load(kpt_dir / f"{real}_keypoints.npy"))
                labels = build_labels(n_frames, segs)

                # 冲突检测: 同一视频在多个 xlsx 中标注不一致
                lf = label_dir / f"{real}_labels.npy"
                if lf.exists():
                    old = np.load(lf)
                    if len(old) == len(labels) and (old != labels).any():
                        conflicts.append(real)
                if not args.dry_run:
                    np.save(lf, labels)
                created.append((real, n_frames, len(segs)))
        wb.close()

    print(f"生成/更新标签: {len(created)} 个")
    for name, n, ns in created[:8]:
        print(f"  {name}: {n}帧, {ns}段")
    if len(created) > 8:
        print(f"  ... 共 {len(created)} 个")
    print(f"跳过(标注名无匹配关键点): {skipped}")
    print(f"冲突(不同xlsx标注不一致): {conflicts}")
    if args.dry_run:
        print("[预览模式] 未写入文件")


if __name__ == "__main__":
    main()
