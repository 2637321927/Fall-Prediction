"""
摄像头实时骨架提取演示
用法: python demo_webcam.py
按 Q 退出
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


def demo_webcam():
    from mmpose.apis import inference_topdown, init_model as init_pose
    from mmdet.apis import init_detector, inference_detector
    from pose_vis import draw_skeleton
    from pose_mm import _get_rtmpose_cfg, RTMPOSE_CKPT, DET_CFG, DET_CKPT

    device = "cuda"
    print("加载人体检测器 (RTMDet-Tiny)...")
    detector = init_detector(DET_CFG, DET_CKPT, device=device)

    print("加载姿态模型 (RTMPose-M)...")
    cfg = _get_rtmpose_cfg()
    pose_model = init_pose(cfg, RTMPOSE_CKPT, device=device)

    print("打开摄像头...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter = []
    frame_count = 0

    print("\n✅ 开始演示! 按 Q 退出\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = cv2.getTickCount()

        # 检测人体
        dets = inference_detector(detector, frame)
        bboxes = []
        if hasattr(dets, 'pred_instances') and len(dets.pred_instances) > 0:
            s = dets.pred_instances.scores.cpu().numpy()
            b = dets.pred_instances.bboxes.cpu().numpy()
            bboxes = [b[i] for i in range(len(s)) if s[i] > 0.3]

        # 姿态估计
        if bboxes:
            try:
                res = inference_topdown(pose_model, frame, np.array(bboxes))
                for r in res:
                    if hasattr(r, 'pred_instances'):
                        kpts = r.pred_instances.keypoints[0].cpu().numpy()
                        scores = r.pred_instances.keypoint_scores[0].cpu().numpy()
                    else:
                        kpts = r['keypoints']; scores = r['keypoint_scores']

                    frame_kpts = np.zeros((17, 3), dtype=np.float32)
                    frame_kpts[:, :2] = kpts
                    frame_kpts[:, 2] = scores
                    frame_kpts[frame_kpts[:, 2] < 0.5] = 0.0

                    frame = draw_skeleton(frame, frame_kpts, 0.5)
            except Exception:
                pass

        # FPS
        t1 = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (t1 - t0)
        fps_counter.append(fps)
        frame_count += 1
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = sum(fps_counter) / len(fps_counter)

        # 显示
        cv2.putText(frame, f"MMPose RTMPose-M | {avg_fps:.0f} FPS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"Detected: {len(bboxes)} person(s)", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Fall Detection Demo - Skeleton Extraction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n演示结束, 共 {frame_count} 帧, 平均 {avg_fps:.1f} FPS")


if __name__ == "__main__":
    demo_webcam()
