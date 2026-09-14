"""
实时跌倒检测 API 服务
用法:
  python server.py --checkpoint ../checkpoints/best_model.pt              # 摄像头
  python server.py --checkpoint ... --video ../video/FDFSM5.mp4          # 视频文件
  python server.py --host 0.0.0.0 --port 8000                            # 监听所有网卡

接口:
  GET  /api/status      # 当前状态快照
  GET  /api/history     # 历史记录
  GET  /api/config      # 当前阈值配置
  POST /api/config      # 修改阈值
  WS   /ws              # WebSocket 实时推送检测结果
  GET  /video           # MJPEG 视频流

架构: 推理线程独立运行, 结果写入共享缓冲区; API 线程异步读取, 互不阻塞
"""
import threading, argparse, time, asyncio, json
from collections import deque
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

import numpy as np


# ============ 共享检测状态 ============
class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {}                          # 最新检测结果
        self.history = deque(maxlen=10000)        # 历史记录
        self.thresholds = {"low": 0.60, "high": 0.75, "immediate": 0.85}
        self.fps = 0.0
        self.online = False
        self.frame = None                         # 最新标注帧(BGR), 供视频流
        self.frame_lock = threading.Lock()

state = DetectionState()


# ============ 预警判断(阈值可动态调整) ============
def get_alert(risk_history, risk, th):
    """根据连续风险分数判断预警等级(规则同状态机, 阈值可配置)"""
    if risk > th["immediate"]:
        return "IMMEDIATE"
    h = list(risk_history)   # deque 不支持切片, 转 list
    if len(h) >= 2 and all(r > th["high"] for r in h[-2:]):
        return "HIGH_RISK"
    if len(h) >= 3 and all(r > th["low"] for r in h[-3:]):
        return "LOW_RISK"
    return "SAFE"


# ============ 推理线程 ============
def inference_worker(checkpoint_path, video_path, device):
    import cv2, torch
    from mmpose.apis import inference_topdown, init_model as init_pose
    from mmdet.apis import init_detector, inference_detector
    from mmengine.registry import DefaultScope
    from stgcn_model import STGCN
    from tcn_model import TCN
    from features import enhance_keypoints
    from pose_vis import draw_skeleton
    from pose_mm import DET_CFG, DET_CKPT, POSE_CFG, POSE_CKPT

    # ---- 加载模型 ----
    print("加载检测器...")
    detector = init_detector(DET_CFG, DET_CKPT, device=device)
    print("加载姿态模型...")
    pose_model = init_pose(POSE_CFG, POSE_CKPT, device=device)

    print("加载跌倒检测模型...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    nc = ckpt.get("num_classes", 2)
    mtype = ckpt.get("model_type", "stgcn")
    model_state = ckpt["model"]
    in_ch = 3
    for k, v in model_state.items():
        if "input_proj" in k and v.dim() == 4:
            in_ch = v.shape[1]
            break
    if mtype == "tcn":
        model = TCN(hidden=cfg.get("hidden", 128), num_layers=cfg.get("num_blocks", 4),
                    num_classes=nc, in_channels=in_ch)
    else:
        model = STGCN(hidden=cfg.get("hidden", 128), num_blocks=cfg.get("num_blocks", 6),
                      num_classes=nc, in_channels=in_ch)
    model.load_state_dict(model_state)
    model.to(device).eval()
    print(f"模型: {mtype.upper()}, F1={ckpt.get('f1', 0):.4f}, in_ch={in_ch}")

    # ---- 视频源 ----
    if video_path:
        cap = cv2.VideoCapture(video_path)
        source = Path(video_path).name
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source = "摄像头"
    print(f"视频源: {source}")

    window = 30
    kpt_buffer = deque(maxlen=window)
    kpt_raw_buffer = deque(maxlen=window)
    risk_history = deque(maxlen=10)
    fps_buf = deque(maxlen=30)
    fidx = 0
    state_names = {0: "NORMAL", 1: "UNSTABLE", 2: "FALLING"}
    state_colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % 20 == 0:
            print(f"[推理] 处理到帧 {fidx}", flush=True)

        kpts_raw = np.zeros((17, 3), dtype=np.float32)
        with DefaultScope.overwrite_default_scope('mmdet'):
            dets = inference_detector(detector, frame)
        bboxes = []
        if hasattr(dets, 'pred_instances') and len(dets.pred_instances) > 0:
            inst = dets.pred_instances
            s = inst.scores.cpu().numpy()
            b = inst.bboxes.cpu().numpy()
            bboxes = [b[i] for i in range(len(s)) if s[i] > 0.3]

        if bboxes:
            res = inference_topdown(pose_model, frame, np.array(bboxes[:1]))
            if len(res) > 0:
                r = res[0]
                if hasattr(r, 'pred_instances'):
                    k = np.array(r.pred_instances.keypoints[0])
                    sc = np.array(r.pred_instances.keypoint_scores[0])
                else:
                    k = r['keypoints']; sc = r['keypoint_scores']
                kpts_raw[:, :2] = k
                kpts_raw[:, 2] = sc
                kpts_raw[kpts_raw[:, 2] < 0.5] = 0.0

        # 归一化: 髋居中 + 除身高
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

        kpts_e = enhance_keypoints(kpts_norm.reshape(1, 17, 3)).reshape(17, in_ch)
        kpt_buffer.append(kpts_e)
        kpt_raw_buffer.append(kpts_raw)

        # 推理
        result = {
            "frame": fidx, "state": "NORMAL", "risk_score": 0.0,
            "alert_level": "SAFE", "probabilities": {"normal": 1.0, "unstable": 0.0, "falling": 0.0},
            "timestamp": time.time(),
        }
        if len(kpt_buffer) == window:
            x = torch.FloatTensor(np.array(list(kpt_buffer))).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0]
            p = probs.cpu().numpy()
            nc_model = len(p)
            if nc_model >= 3:
                risk = float(p[1] + p[2])
                state_idx = int(np.argmax(p))
            else:
                risk = float(p[1])
                state_idx = 0
            with state.lock:
                th = dict(state.thresholds)
            risk_history.append(risk)
            alert = get_alert(risk_history, risk, th)
            result = {
                "frame": fidx,
                "state": state_names.get(state_idx, "UNKNOWN"),
                "risk_score": round(risk, 4),
                "alert_level": alert,
                "probabilities": {
                    "normal": round(float(p[0]), 4) if nc_model >= 3 else round(float(p[0]), 4),
                    "unstable": round(float(p[1]), 4) if nc_model >= 3 else 0.0,
                    "falling": round(float(p[2]), 4) if nc_model >= 3 else 0.0,
                },
                "timestamp": time.time(),
            }
            # 画标注帧(供视频流)
            disp = frame.copy()
            draw_skeleton(disp, kpts_raw, 0.5)
            color = state_colors.get(state_idx, (255, 255, 255))
            hh, ww = disp.shape[:2]
            bar_w = int(risk * ww)
            cv2.rectangle(disp, (0, hh - 40), (bar_w, hh - 25), color, -1)
            cv2.rectangle(disp, (0, hh - 40), (ww, hh - 25), (255, 255, 255), 1)
            cv2.putText(disp, f"{result['state']} ({risk:.2f}) {alert}", (10, hh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(disp, f"Frame:{fidx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            with state.frame_lock:
                state.frame = disp

        # 更新共享状态
        with state.lock:
            state.latest = result
            state.history.append({"frame": fidx, **result})
            state.online = True
        fps_buf.append(1 / (time.time() - t0 + 1e-3))
        with state.lock:
            state.fps = float(sum(fps_buf) / len(fps_buf))

        fidx += 1

    with state.lock:
        state.online = False
    cap.release()
    print("推理线程结束")


# ============ FastAPI 服务 ============
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

app = FastAPI(title="Fall Detection API")


@app.get("/api/status")
def api_status():
    with state.lock:
        return {"online": state.online, "fps": round(state.fps, 2),
                "thresholds": state.thresholds, **state.latest}


@app.get("/api/history")
def api_history(limit: int = 100):
    with state.lock:
        recs = list(state.history)[-limit:]
        return {"total": len(state.history), "records": recs}


@app.get("/api/config")
def api_get_config():
    with state.lock:
        return state.thresholds


@app.post("/api/config")
def api_set_config(body: dict):
    with state.lock:
        for k in ["low", "high", "immediate"]:
            if k in body:
                state.thresholds[k] = float(body[k])
        return {"ok": True, "thresholds": state.thresholds}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with state.lock:
                data = dict(state.latest)
                data["online"] = state.online
                data["fps"] = round(state.fps, 2)
            try:
                await ws.send_json(data)
            except Exception:
                break
            await asyncio.sleep(0.5)   # 每0.5秒推送一次
    except WebSocketDisconnect:
        pass


def gen_frames():
    """MJPEG 视频流: 无人订阅时主循环只 sleep, 不消耗编码资源"""
    while True:
        with state.frame_lock:
            frame = state.frame
        if frame is not None:
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.get("/video")
def api_video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def main():
    p = argparse.ArgumentParser(description="实时跌倒检测 API 服务")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", type=str, default=None, help="视频文件(默认摄像头)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    import cv2  # 提前 import, 供 gen_frames 闭包使用

    # 启动推理线程(独立, 不阻塞 API)
    t = threading.Thread(target=inference_worker,
                         args=(args.checkpoint, args.video, args.device),
                         daemon=True)
    t.start()

    import uvicorn
    print(f"\nAPI 服务启动: http://{args.host}:{args.port}")
    print("  GET  /api/status    当前状态")
    print("  GET  /api/history   历史记录")
    print("  GET  /api/config    阈值配置")
    print("  POST /api/config    修改阈值")
    print("  WS   /ws            实时推送")
    print("  GET  /video         视频流\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
