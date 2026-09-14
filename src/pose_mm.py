"""
模块A: MMPose/RTMPose 提取 17 个 COCO 关键点 + 输出标注视频
用法: python pose_mm.py --video test.mp4
输出: test_s.mp4 (标注骨架) + test_keypoints.npy
"""

import numpy as np
from pathlib import Path
import argparse


def _find_config(pkg_name, pattern):
    """在 mm 包的 .mim/configs 中查找配置文件"""
    import importlib, glob, os
    mod = importlib.import_module(pkg_name)
    base = os.path.join(os.path.dirname(mod.__file__), '.mim', 'configs')
    matches = glob.glob(os.path.join(base, '**', pattern), recursive=True)
    return matches[0] if matches else None


# 自动定位模型配置（首次运行会下载权重）
DET_CFG = _find_config('mmdet', 'rtmdet_tiny_8xb32-300e_coco.py')
DET_CKPT = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/" \
           "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

# 模型配置与权重（首次需运行: mim download mmpose --config rtmpose-m_8xb256-420e_coco-256x192 --dest checkpoints）
# 路径基于脚本位置解析，从任意工作目录运行都能找到
_SRC_DIR = Path(__file__).resolve().parent
POSE_CFG = str(_SRC_DIR / ".." / "configs" / "rtmpose-m_8xb256-420e_coco-256x192.py")
POSE_CKPT = str(_SRC_DIR / ".." / "checkpoints" /
                "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth")


def load_models(device="cuda"):
    """加载检测器+姿态模型（只调用一次）"""
    from mmengine.registry import DefaultScope
    from mmpose.apis import init_model as init_pose
    from mmdet.apis import init_detector

    print(f"加载检测器: {DET_CFG}")
    detector = init_detector(DET_CFG, DET_CKPT, device=device)
    print(f"加载姿态模型: {POSE_CFG}")
    pose_model = init_pose(POSE_CFG, POSE_CKPT, device=device)
    return detector, pose_model


def extract_keypoints(video_path, target_fps=0, conf_threshold=0.5,
                      save_dir=None, show_video=False, device="cuda",
                      detector=None, pose_model=None, save_video=True):
    """
    提取关键点 + 可选输出标注视频（*_s.mp4）
    detector/pose_model 可传预加载的模型，避免重复加载
    save_video=False 可跳过标注视频生成，大幅提速
    返回: (N_frames, 17, 3) numpy 数组
    """
    import cv2
    from mmengine.registry import DefaultScope
    from mmpose.apis import inference_topdown
    from mmdet.apis import inference_detector
    from pose_vis import draw_skeleton

    models_loaded = detector is not None and pose_model is not None
    if detector is None or pose_model is None:
        detector, pose_model = load_models(device)

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 抽帧
    interval = max(1, int(orig_fps / target_fps)) if target_fps and orig_fps > target_fps else 1

    if save_video:
        out_name = video_path.stem + "_s.mp4"
        out_path = str(video_path.parent / out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = orig_fps / interval
        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))
    else:
        writer = None

    print(f"视频: {video_path.name} ({width}x{height}, {orig_fps:.1f}fps"
          + (f" → {out_fps:.1f}fps" if interval > 1 else "") + ")")
    if save_video:
        print(f"输出: {out_name}")

    keypoints_all = []
    fidx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 抽帧：跳过不需要处理的帧
        if fidx % interval != 0:
            if writer: writer.write(frame)
            fidx += 1; continue

        # 人体检测（切回 mmdet scope）
        with DefaultScope.overwrite_default_scope('mmdet'):
            dets = inference_detector(detector, frame)
        bboxes = []
        if hasattr(dets, 'pred_instances') and len(dets.pred_instances) > 0:
            inst = dets.pred_instances
            s = inst.scores.cpu().numpy()
            b = inst.bboxes.cpu().numpy()
            bboxes = [b[i] for i in range(len(s)) if s[i] > 0.3]

        # 姿态估计
        frame_kpts = np.zeros((17, 3), dtype=np.float32)
        if bboxes:
            res = inference_topdown(pose_model, frame, np.array(bboxes[:1]))
            if len(res) > 0:
                r = res[0]
                if hasattr(r, 'pred_instances'):
                    k = np.array(r.pred_instances.keypoints[0])
                    sc = np.array(r.pred_instances.keypoint_scores[0])
                else:
                    k = r['keypoints']; sc = r['keypoint_scores']
                frame_kpts[:, :2] = k; frame_kpts[:, 2] = sc
                frame_kpts[frame_kpts[:, 2] < conf_threshold] = 0.0
                # 画骨架（仅生成视频时）
                if writer:
                    frame = draw_skeleton(frame, frame_kpts, conf_threshold)
                # 归一化：髋居中 + 除身高 → 平移+缩放不变
                hip_x = (frame_kpts[11, 0] + frame_kpts[12, 0]) / 2
                hip_y = (frame_kpts[11, 1] + frame_kpts[12, 1]) / 2
                head_y = frame_kpts[0, 1]
                ankle_y = max(frame_kpts[15, 1], frame_kpts[16, 1])
                h = ankle_y - head_y
                if h > 20:
                    frame_kpts[:, 0] -= hip_x
                    frame_kpts[:, 1] -= hip_y
                    frame_kpts[:, :2] /= h

        keypoints_all.append(frame_kpts); fidx += 1

        if writer:
            cv2.putText(frame, f"Frame:{fidx} Persons:{len(bboxes)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            writer.write(frame)

        if show_video:
            cv2.imshow("RTMPose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                show_video = False; cv2.destroyAllWindows()

        if fidx % 50 == 0:
            print(f"  {fidx} 帧...")

    cap.release()
    if writer: writer.release()
    if show_video: cv2.destroyAllWindows()

    keypoints_all = np.array(keypoints_all, dtype=np.float32)
    print(f"完成: {keypoints_all.shape[0]} 帧骨架" + (f" → {out_path}" if writer else ""))

    if save_dir:
        d = Path(save_dir); d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"{video_path.stem}_keypoints.npy", keypoints_all)
        print(f"关键点: {d / f'{video_path.stem}_keypoints.npy'}")

    return keypoints_all


def main():
    p = argparse.ArgumentParser(description="MMPose/RTMPose 提取关键点 + 生成标注视频")
    p.add_argument("--video", required=True, help="视频路径")
    p.add_argument("--fps", type=int, default=0, help="目标帧率，0=每帧提取")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--save_dir", type=str, default="../data/keypoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--show", action="store_true")
    p.add_argument("--no_video", action="store_true", help="不生成标注视频，大幅提速")
    a = p.parse_args()
    extract_keypoints(a.video, a.fps, a.conf, a.save_dir, a.show, a.device,
                      save_video=not a.no_video)


if __name__ == "__main__":
    main()
