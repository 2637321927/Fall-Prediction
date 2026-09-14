"""
推理脚本: MMPose + TCN/ST-GCN + 状态机 全链路跌倒检测
用法: python infer.py --checkpoint ../checkpoints/best_model.pt --input test.npy
"""

import torch, numpy as np, argparse
from pathlib import Path

from tcn_model import TCN
from stgcn_model import STGCN
from state_machine import FallStateMachine, AlertLevel


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mtype = ckpt.get("model_type", "stgcn")
    nc = ckpt.get("num_classes", 2)

    # 从权重形状反推输入通道数
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

    model.load_state_dict(state)
    model.to(device).eval()
    print(f"模型: {mtype.upper()} (ep {ckpt.get('epoch','?')}, F1={ckpt.get('f1',0):.4f}, in_ch={in_ch})")
    return model, in_ch


def infer_keypoints(model, npy_path, in_ch=3, window=16, stride=1, device="cuda"):
    """从 .npy 关键点文件推理"""
    kpts = np.load(npy_path)
    if in_ch > 3:
        from features import enhance_keypoints
        kpts = enhance_keypoints(kpts)
    fsm = FallStateMachine()
    scores, alerts = [], []

    for i in range(0, len(kpts) - window + 1, stride):
        x = torch.FloatTensor(kpts[i:i + window]).unsqueeze(0).to(device)
        with torch.no_grad():
            s = model.predict_risk(x).item()
        scores.append(s)
        alerts.append(fsm.update(s))

    return np.array(scores), alerts


def infer_video(model, video_path, in_ch=3, window=16, stride=1, device="cuda",
                show=False, save=False):
    """实时推理: MMPose逐帧提取关键点 → TCN/ST-GCN判断 → 画面显示预警等级"""
    import cv2
    from pose_mm import extract_keypoints
    from pose_vis import draw_skeleton

    print("Step 1: 提取关键点...")
    kpts = extract_keypoints(video_path, show_video=False, save_video=False, device=device)

    if in_ch > 3:
        from features import enhance_keypoints
        kpts = enhance_keypoints(kpts)

    print(f"Step 2: 推理 {len(kpts)} 帧...")
    fsm = FallStateMachine()
    results = []

    for i in range(0, len(kpts) - window + 1, stride):
        x = torch.FloatTensor(kpts[i:i + window]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            nc = logits.shape[1]
            if nc >= 3:
                score = probs[1:].sum().item()
            else:
                score = probs[1].item()
        level = fsm.update(score)
        results.append({"frame": i, "risk_score": score, "alert": level.value,
                        "probs": probs.cpu().tolist()})

    # 打印摘要
    alerts = [r for r in results if r["alert"] != "SAFE"]
    print(f"\n总窗口: {len(results)}, 预警: {len(alerts)}")
    print(f"前5帧概率分布 (class0=正常, class1+2=风险):")
    for r in results[:5]:
        print(f"  帧{r['frame']:5d}: probs={r['probs']} → score={r['risk_score']:.3f} {r['alert']}")
    for a in alerts[:10]:
        print(f"  帧{a['frame']:5d}: score={a['risk_score']:.3f} → {a['alert']}")
    if len(alerts) > 10:
        print(f"  ... 共 {len(alerts)} 条预警\n")

    # 渲染标注帧（骨架 + 预警条）
    if show or save:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save:
            result_dir = Path("../result"); result_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(result_dir / (Path(video_path).stem + "_result.mp4"))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"保存结果视频: {Path(out_path).name}")

        if show:
            cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Fall Detection", min(w, 1280), min(h, 720))
        if show or save:
            print("=" * 40)

        alert_idx = 0
        frame_idx = 0
        colors = {"SAFE": (0, 255, 0), "LOW_RISK": (0, 255, 255),
                  "HIGH_RISK": (0, 165, 255), "IMMEDIATE": (0, 0, 255)}
        # 三分类具体状态（cv2默认字体不支持中文，用英文）
        state_names = {0: "NORMAL", 1: "UNSTABLE", 2: "FALLING"}
        state_colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 画骨架（frame_idx 匹配 extract_keypoints 的帧序）
            if frame_idx < len(kpts):
                frame = draw_skeleton(frame, kpts[frame_idx, :, :3], 0.5)

            # 查当前帧的预警等级
            while alert_idx < len(results) and results[alert_idx]["frame"] <= frame_idx:
                alert_idx += 1
            current = results[alert_idx - 1] if alert_idx > 0 else None

            if current:
                risk = current["risk_score"]
                level = current["alert"]
                probs = current["probs"]

                # 判断模型认为的具体状态（argmax）
                if len(probs) >= 3:
                    state_idx = int(np.argmax(probs))
                    state = state_names.get(state_idx, level)
                    color = state_colors.get(state_idx, (255, 255, 255))
                else:
                    state = level
                    color = colors.get(level, (255, 255, 255))

                bar_w = int(risk * w)
                cv2.rectangle(frame, (0, h - 40), (bar_w, h - 25), color, -1)
                cv2.rectangle(frame, (0, h - 40), (w, h - 25), (255, 255, 255), 1)
                cv2.putText(frame, f"{state} ({risk:.2f})", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("Fall Detection", frame)
                key = cv2.waitKey(25) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return results


def main():
    p = argparse.ArgumentParser(description="跌倒检测推理")
    p.add_argument("--checkpoint", required=True, help="模型权重路径")
    p.add_argument("--input", required=True, help="输入 .npy 或视频文件")
    p.add_argument("--mode", default="keypoints", choices=["keypoints", "video"])
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", action="store_true", help="保存标注结果视频 (*_result.mp4)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, in_ch = load_model(args.checkpoint, str(device))

    if args.mode == "keypoints":
        scores, alerts = infer_keypoints(model, args.input, in_ch, args.window, args.stride, str(device))
        print(f"\n完成! 最高风险: {scores.max():.4f}, 平均: {scores.mean():.4f}")
        from collections import Counter
        print(f"预警统计: {dict(Counter(a.value for a in alerts))}")
    else:
        infer_video(model, args.input, in_ch, args.window, args.stride, str(device),
                    args.show, args.save)


if __name__ == "__main__":
    main()
