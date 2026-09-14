"""
批量检测视频: 对 video 目录下每个视频跑跌倒检测推理, 汇总输出每段结果
用法: python detect_videos.py [--video_dir ../video] [--checkpoint ../checkpoints/best_model.pt]
"""
import cv2, torch, numpy as np, argparse, time, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))


_SRC_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir", type=str, default=str(_SRC_DIR / ".." / "video"))
    p.add_argument("--checkpoint", type=str, default=str(_SRC_DIR / ".." / "checkpoints" / "best_model.pt"))
    p.add_argument("--result_dir", type=str, default=str(_SRC_DIR / ".." / "result"))
    p.add_argument("--infer_fps", type=float, default=15.0)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--detect_every", type=int, default=5)
    p.add_argument("--infer_width", type=int, default=960)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ext", type=str, default="mp4")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    from mmdet.apis import init_detector
    from mmpose.apis import init_model as init_pose
    from pose_mm import DET_CFG, DET_CKPT, POSE_CFG, POSE_CKPT
    from fast_infer import make_fast_det_infer, make_fast_pose_infer
    from stgcn_model import STGCN
    from tcn_model import TCN
    from state_machine import FallStateMachine
    from features import enhance_keypoints

    print("加载检测器...")
    detector = init_detector(DET_CFG, DET_CKPT, device=device)
    print("加载姿态模型...")
    pose_model = init_pose(POSE_CFG, POSE_CKPT, device=device)
    det_infer = make_fast_det_infer(detector)
    pose_infer = make_fast_pose_infer(pose_model)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    nc = ckpt.get("num_classes", 2)
    mtype = ckpt.get("model_type", "stgcn")
    state = ckpt["model"]
    in_ch = 3
    for k, v in state.items():
        if "input_proj" in k and v.dim() == 4:
            in_ch = v.shape[1]
            break
    if mtype == "tcn":
        model = TCN(hidden=cfg.get("hidden", 128), num_layers=cfg.get("num_blocks", 4),
                    num_classes=nc, in_channels=in_ch)
    else:
        model = STGCN(hidden=cfg.get("hidden", 128), num_blocks=cfg.get("num_blocks", 6),
                      num_classes=nc, in_channels=in_ch)
    model.load_state_dict(state); model.to(device).eval()
    print(f"模型: {mtype.upper()}, in_ch={in_ch}, nc={nc}\n")

    from pose_vis import draw_skeleton
    colors = {"SAFE": (0, 255, 0), "LOW_RISK": (0, 255, 255),
              "MID_RISK": (0, 165, 255), "HIGH_RISK": (0, 100, 255),
              "IMMEDIATE": (0, 0, 255)}

    sample_interval = 1.0 / max(args.infer_fps, 1e-6)
    vids = sorted(Path(args.video_dir).glob(f"*.{args.ext}"))
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果视频目录: {result_dir}")
    print(f"待检测视频: {len(vids)} 个\n{'='*70}")

    for vf in vids:
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            print(f"{vf.name}: 无法打开, 跳过")
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = result_dir / f"{vf.stem}_result.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                 src_fps, (W, H))
        if not writer.isOpened():
            print(f"{vf.name}: 无法创建输出视频, 跳过")
            cap.release(); continue

        norm_buffer = []
        kpt_buffer = []
        fsm = FallStateMachine()
        last_sample_t = time.time() - sample_interval
        sample_cnt = 0
        window_cnt = 0
        probs_list = []
        level_count = {"SAFE": 0, "LOW_RISK": 0, "MID_RISK": 0, "HIGH_RISK": 0, "IMMEDIATE": 0}
        alert_segments = []   # (start, end, level)
        cur_level = None
        cur_start = 0
        bboxes = []
        fidx = 0
        # 叠加状态: 非采样帧沿用上一采样帧的绘制信息, 保持输出视频流畅
        overlay = {"risk": 0.0, "alert": "SAFE", "color": (0, 255, 0),
                   "kpts": np.zeros((17, 3), dtype=np.float32), "boxes": []}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fidx += 1
            if (time.time() - last_sample_t) >= sample_interval:
                last_sample_t += sample_interval
                while last_sample_t + sample_interval <= time.time():
                    last_sample_t += sample_interval
                sample_cnt += 1

                # 检测
                if sample_cnt % args.detect_every == 0 or not bboxes:
                    fh, fw = frame.shape[:2]
                    if args.infer_width > 0:
                        df = cv2.resize(frame, (args.infer_width, int(fh * args.infer_width / fw)))
                        boxes, scores = det_infer(df)
                        sx, sy = fw / df.shape[1], fh / df.shape[0]
                        bboxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
                                  for i, b in enumerate(boxes) if scores[i] > 0.3]
                    else:
                        boxes, scores = det_infer(frame)
                        bboxes = [boxes[i] for i in range(len(scores)) if scores[i] > 0.3]
                else:
                    ex = []
                    fw, fh = frame.shape[1], frame.shape[0]
                    for b in bboxes:
                        x1, y1, x2, y2 = b
                        w, h = x2 - x1, y2 - y1
                        ex.append([max(0, x1-0.1*w), max(0, y1-0.1*h), min(fw, x2+0.1*w), min(fh, y2+0.1*h)])
                    bboxes = ex

                kpts_raw = np.zeros((17, 3), dtype=np.float32)
                if bboxes:
                    res = pose_infer(frame, np.array(bboxes[:1]))
                    if len(res) > 0:
                        r = res[0]
                        k = np.array(r.keypoints[0]); sc = np.array(r.keypoint_scores[0])
                        kpts_raw[:, :2] = k; kpts_raw[:, 2] = sc
                        kpts_raw[kpts_raw[:, 2] < 0.5] = 0.0
                # 归一化
                kpts_norm = kpts_raw.copy()
                hip_x = (kpts_norm[11, 0] + kpts_norm[12, 0]) / 2
                hip_y = (kpts_norm[11, 1] + kpts_norm[12, 1]) / 2
                head_y = kpts_norm[0, 1]
                ankle_y = max(kpts_norm[15, 1], kpts_norm[16, 1])
                h = ankle_y - head_y
                if h > 20:
                    kpts_norm[:, 0] -= hip_x
                    kpts_norm[:, 1] -= hip_y
                    kpts_norm[:, :2] /= h
                norm_buffer.append(kpts_norm)
                if in_ch > 3:
                    prev = norm_buffer[-2] if len(norm_buffer) >= 2 else kpts_norm
                    kpts_e = enhance_keypoints(np.stack([prev, kpts_norm]))[-1]
                else:
                    kpts_e = kpts_norm
                kpt_buffer.append(kpts_e)
                overlay["kpts"] = kpts_raw.copy()
                overlay["boxes"] = [[int(x) for x in b] for b in bboxes]

                if len(kpt_buffer) >= args.window:
                    kpt_buffer = kpt_buffer[-args.window:]
                    x = torch.FloatTensor(np.array(kpt_buffer)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(model(x), dim=-1)[0].cpu().numpy()
                    level = fsm.update(probs if nc >= 3 else probs[1]).value
                    overlay["alert"] = level
                    overlay["color"] = colors.get(level, (255, 255, 255))
                    overlay["risk"] = float(probs[1:].sum()) if nc >= 3 else float(probs[1])
                    window_cnt += 1
                    probs_list.append(probs)
                    level_count[level] = level_count.get(level, 0) + 1
                    if level != cur_level:
                        if cur_level is not None and cur_level != "SAFE":
                            alert_segments.append((cur_start, window_cnt, cur_level))
                        cur_level = level
                        cur_start = window_cnt

            # 绘制叠加并写入输出视频
            draw_skeleton(frame, overlay["kpts"], 0.5)
            for x1, y1, x2, y2 in overlay["boxes"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            h, w = frame.shape[:2]
            # 风险条
            cv2.rectangle(frame, (0, h - 40), (int(overlay["risk"] * w), h - 25), overlay["color"], -1)
            cv2.rectangle(frame, (0, h - 40), (w, h - 25), (255, 255, 255), 1)
            cv2.putText(frame, f"{overlay['alert']} ({overlay['risk']:.2f})", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay["color"], 2)
            cv2.putText(frame, f"Frame:{fidx}", (w - 130, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)

        if cur_level is not None and cur_level != "SAFE":
            alert_segments.append((cur_start, window_cnt, cur_level))
        cap.release()
        writer.release()
        print(f"  结果视频: {out_path}")

        # 统计
        if probs_list:
            probs_mean = np.mean(probs_list, axis=0)
            p0, p1, p2 = probs_mean[0], (probs_mean[1] if nc > 1 else 0), (probs_mean[2] if nc > 2 else 0)
        else:
            p0 = p1 = p2 = 0.0
        highest = [lv for lv in ["IMMEDIATE", "HIGH_RISK", "MID_RISK", "LOW_RISK"] if level_count[lv] > 0]
        highest = highest[0] if highest else "SAFE"
        seg_str = ", ".join(f"窗口{w0}-{w1}:{lv}" for w0, w1, lv in alert_segments) if alert_segments else "无"
        print(f"\n[{vf.name}]")
        print(f"  视频: {total_frames}帧({src_fps:.0f}fps) | 采样窗口: {window_cnt}个")
        print(f"  平均概率: 正常={p0:.2f} 不正常={p1:.2f} 摔倒={p2:.2f}")
        print(f"  最高报警: {highest} | 各等级窗口: { {k: level_count[k] for k in ['LOW_RISK','MID_RISK','HIGH_RISK','IMMEDIATE']} }")
        print(f"  报警时段: {seg_str}")
        print(f"  结论: {('检测到摔倒/高危' if highest in ('IMMEDIATE','HIGH_RISK') else ('存在风险' if highest == 'MID_RISK' else ('低风险' if highest == 'LOW_RISK' else '未检测到异常')))}")

    print(f"\n{'='*70}\n完成, 共检测 {len(vids)} 个视频")


if __name__ == "__main__":
    main()
