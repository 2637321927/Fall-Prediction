"""
URFD 视频裁切: 去掉左半深度图，只保留右半 RGB
用法: python crop_urfd.py --input fall-01-cam0-rgb.avi --output fall-01.mp4
      python crop_urfd.py --all  (批量处理 fall/urfd/ 下所有视频)
"""

import cv2
from pathlib import Path
import argparse


def crop_video(input_path, output_path=None):
    """裁切视频：只保留右半部分"""
    inp = Path(input_path)
    if output_path is None:
        output_path = str(inp.parent / f"{inp.stem}_crop{inp.suffix}")
    
    cap = cv2.VideoCapture(str(inp))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    new_w = w // 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_w, h))
    
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(frame[:, w//2:, :])
        cnt += 1
        if cnt % 100 == 0: print(f"  {cnt}/{total}")
    
    cap.release(); out.release()
    print(f"  {inp.name} → {Path(output_path).name} ({new_w}x{h}, {cnt}帧)")


def main():
    p = argparse.ArgumentParser(description="URFD 视频裁切（去左半深度图）")
    p.add_argument("--input", type=str, help="单个视频路径")
    p.add_argument("--output", type=str, default=None, help="输出路径（可选）")
    p.add_argument("--all", action="store_true", help="批量处理 data/raw/urfd/ 下所有视频")
    p.add_argument("--dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "urfd"),
                   help="批量处理的目录, 默认 <项目>/data/raw/urfd")
    a = p.parse_args()
    
    if a.all:
        d = Path(a.dir)
        videos = sorted(d.glob("*.avi")) + sorted(d.glob("*.mp4"))
        print(f"找到 {len(videos)} 个视频\n")
        for vf in videos:
            crop_video(str(vf))
        print(f"\n完成 {len(videos)} 个")
    elif a.input:
        crop_video(a.input, a.output)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
