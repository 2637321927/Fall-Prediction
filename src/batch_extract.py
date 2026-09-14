"""
批量关键点提取（模型只加载一次）
输出按源目录层级镜像到 --save_dir: clips/xxx.avi → <save_dir>/clips/xxx_keypoints.npy
用法: python batch_extract.py                       # 提取 data/raw 下全部视频
      python batch_extract.py --dir ../data/raw/urfd  # 只提取 URFD
      python batch_extract.py --dir ../data/raw/clips --exts avi  # 自定义扩展名
"""

import argparse, time, sys
from pathlib import Path

# 确保能 import 同目录的 pose_mm
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pose_mm import load_models, extract_keypoints

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=str(_PROJECT_ROOT / "data" / "raw"),
                    help="视频目录(会递归查找子目录), 默认 <项目>/data/raw")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--fps", type=int, default=0, help="0=每帧提取; 例如 16=每秒取16帧")
parser.add_argument("--conf", type=float, default=0.5)
parser.add_argument("--save_dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "data" / "keypoints"),
                    help="输出根目录, 默认 <项目>/data/keypoints (与 src 同级)")
parser.add_argument("--exts", type=str, default="mp4",
                    help="要处理的视频扩展名(逗号分隔); 默认只处理 mp4, 其余(含 avi)跳过")
args = parser.parse_args()

fall_dir = Path(args.dir).resolve()
exts = tuple(e.strip().lstrip(".").lower() for e in args.exts.split(",") if e.strip())
video_files = sorted(f for f in fall_dir.rglob("*") if f.suffix.lower().lstrip(".") in exts)
total = len(video_files)
print(f"目录: {fall_dir}")
print(f"扩展名: {exts}")
print(f"共 {total} 段视频\n")
print("=" * 40)
detector, pose_model = load_models(args.device)
print("=" * 40 + "\n")

failed = []
t_start = time.time()
out_dir = Path(args.save_dir)
for i, vf in enumerate(video_files):
    # 镜像源目录层级: clips/xxx.avi → <save_dir>/clips/xxx_keypoints.npy
    rel_dir = vf.parent.relative_to(fall_dir) if vf.parent != fall_dir else Path(".")
    sub_dir = out_dir / rel_dir
    rel_name = f"{vf.parent.name}/{vf.name}"
    print(f"[{i+1}/{total}] {rel_name} → {sub_dir}")
    t0 = time.time()
    try:
        kpts = extract_keypoints(
            str(vf), target_fps=args.fps, conf_threshold=args.conf,
            save_dir=str(sub_dir), show_video=False, device=args.device,
            detector=detector, pose_model=pose_model,
            save_video=False,  # 批量提取不生成视频
        )
        elapsed = time.time() - t0
        print(f"  ✅ 关键点: {kpts.shape} → {sub_dir / f'{vf.stem}_keypoints.npy'} ({elapsed:.0f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        failed.append(str(vf))
        print(f"  ❌ 失败 ({elapsed:.0f}s): {e}")
    print()

t_total = time.time() - t_start
print(f"\n{'='*40}")
print(f"完成: {total - len(failed)}/{total} 成功  总耗时: {t_total:.0f}s")
if failed:
    print(f"失败 ({len(failed)}):")
    for f in failed:
        print(f"  {f}")
