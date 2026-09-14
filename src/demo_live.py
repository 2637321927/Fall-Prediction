"""
实时跌倒检测 (摄像头 / 视频文件 / RTMP 网络流)
用法:
  python demo_live.py --checkpoint ../checkpoints/best_model.pt                        # 摄像头
  python demo_live.py --checkpoint ../checkpoints/best_model.pt --video test.mp4      # 视频文件
  python demo_live.py --checkpoint ../checkpoints/best_model.pt --video "rtmp://..."  # RTMP 网络流
默认每帧提取骨架并推理（预处理与训练完全一致），滑动窗口判风险，实时叠加画面
网络流断线会自动重连
按 Q 退出

注意: 网络流地址含 & ? = 等特殊字符时，整条地址必须用引号包起来，
否则会被 shell 截断（如 PowerShell 中 & 会被当成命令分隔符）。
"""

import cv2, torch, numpy as np, argparse, time, subprocess
from collections import deque
from pathlib import Path
import sys; sys.path.append(str(Path(__file__).parent))


def _pick_port():
    """找一个空闲的本地端口"""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_ffmpeg_relay(url, out_width=0):
    """用 ffmpeg 拉流转成 H.264，通过本地 tcp 端口输出；out_width>0 时强制缩放宽度
    返回 (子进程, OpenCV 可读的本地地址)"""
    import imageio_ffmpeg
    port = _pick_port()
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y",
           "-fflags", "nobuffer", "-flags", "low_delay",
           "-analyzeduration", "3000000", "-probesize", "3000000",
           "-i", url,
           "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
           "-an", "-f", "flv", f"tcp://127.0.0.1:{port}?listen"]
    if out_width > 0:
        # 强制缩放到指定宽度(-2 保证高度为偶数), 大幅降低码率/带宽
        idx = cmd.index("-c:v")
        cmd[idx:idx] = ["-vf", f"scale={out_width}:-2"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, f"tcp://127.0.0.1:{port}"


def _open_source(src, allow_relay=False, force_relay_width=0):
    """打开视频源（文件或网络流），对网络流设置超时避免读帧卡死
    allow_relay=True 且直接打开失败时，自动用 ffmpeg 转码回退(H265→H264)
    force_relay_width>0 时强制走 ffmpeg 转码并缩放到该宽度(缓解网络读帧慢)
    返回 (cap, relay_proc)"""
    cap = cv2.VideoCapture(src)
    # 仅 FFmpeg 后端生效的属性，防止网络异常时 read() 永久阻塞
    for prop in (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cv2.CAP_PROP_READ_TIMEOUT_MSEC):
        try:
            cap.set(prop, 15000)
        except Exception:
            pass

    relay = None
    use_relay = (not cap.isOpened() and allow_relay) or force_relay_width > 0
    if use_relay:
        if cap.isOpened():
            cap.release()
        print("直接打开失败（可能是 H.265 流）→ 启动 ffmpeg 转码为 H.264 ..." if not cap.isOpened()
              else f"强制 ffmpeg 转码缩放 (宽 {force_relay_width}) 以缓解网络读帧慢 ...")
        relay, local_url = _start_ffmpeg_relay(src, force_relay_width)
        time.sleep(3.0)  # 等 ffmpeg 连上源流
        cap = cv2.VideoCapture(local_url)
        for prop in (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, cv2.CAP_PROP_READ_TIMEOUT_MSEC):
            try:
                cap.set(prop, 15000)
            except Exception:
                pass
        if not cap.isOpened():
            cap.release()
            relay.terminate()
            relay = None

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频源: {src}\n"
            "若是网络流: 1) 确认 opencv 带 FFmpeg 支持; 2) 含 & 的地址必须加引号; "
            "3) H265 流解码失败可把地址中 supportH265=1 改为 0")
    return cap, relay


def demo_live(checkpoint_path, video_path=None, extract_every=1, window=30,
              device="cuda", display_width=960, play_only=False, detect_every=3,
              profile=False, infer_fps=15.0, infer_width=0, relay_width=0):
    """
    infer_fps: 训练数据统一后的帧率(默认15)。推理时按该帧率时间采样进窗口,
               与训练数据的时间重采样严格一致(25fps/30fps 源流都能均匀降到目标帧率)。
    infer_width: 推理检测分辨率宽度(0=原图)。2K/高清流建议设 960/640, 检测大幅提速;
                bbox 自动还原回原图坐标, 姿态/骨架仍用原图, 不影响精度。
    relay_width: 强制用 ffmpeg 把网络流转码并缩放到该宽度(>0)。网络读帧慢时大幅降码率,
                缓解读帧等待; 0=不强制(仅打不开时回退)。
    """
    in_ch, nc = 3, 2
    if play_only:
        detector = pose_model = model = fsm = None
        print("play_only 模式：只播放视频流，不加载模型、不推理\n")
    else:
        from mmpose.apis import inference_topdown, init_model as init_pose
        from mmdet.apis import init_detector, inference_detector
        from mmengine.registry import DefaultScope
        from stgcn_model import STGCN
        from tcn_model import TCN
        from state_machine import FallStateMachine
        from pose_mm import DET_CFG, DET_CKPT, POSE_CFG, POSE_CKPT

        # 加载模型
        print("加载检测器...")
        detector = init_detector(DET_CFG, DET_CKPT, device=device)
        print("加载姿态模型...")
        pose_model = init_pose(POSE_CFG, POSE_CKPT, device=device)

        print("加载跌倒检测模型...")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        nc = ckpt.get("num_classes", 2)
        mtype = ckpt.get("model_type", "stgcn")
        # 从权重反推输入通道数
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
        fsm = FallStateMachine()
        print(f"模型: {mtype.upper()}, F1={ckpt.get('f1', 0):.4f}, in_ch={in_ch}\n")

        # 手写紧凑后处理：绕开 mmdet/mmpose 的 test_step 每帧 Python 包装开销
        from fast_infer import make_fast_det_infer, make_fast_pose_infer
        det_infer = make_fast_det_infer(detector)
        pose_infer = make_fast_pose_infer(pose_model)
    is_url = bool(video_path) and video_path.lower().startswith(
        ("rtmp://", "rtsp://", "http://", "https://"))
    if video_path:
        cap, relay = _open_source(video_path, allow_relay=is_url, force_relay_width=relay_width)
        source_name = video_path[:80] if is_url else Path(video_path).name
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_name = "摄像头"
        relay = None
    print(f"视频源: {source_name}，按 Q 退出\n")
    fps_stream = cap.get(cv2.CAP_PROP_FPS)
    if not fps_stream or fps_stream <= 0:
        fps_stream = 25.0  # 网络流 FPS 常不可靠，按 25fps 估算
    proc_ms_last = 0.0     # 上一帧处理耗时(ms)，用于判断是否需要清积压

    # 滑动窗口缓冲区
    norm_buffer = deque(maxlen=window)      # 归一化关键点序列（用于按训练方式计算帧间速度等增强特征）
    kpt_buffer = deque(maxlen=window)       # 增强后的关键点（供模型）
    kpt_buffer_raw = deque(maxlen=window)   # 原始像素关键点（供画图）
    fps_buffer = deque(maxlen=30)
    fidx = 0
    sample_cnt = 0                       # 实际进入窗口的采样帧计数
    sample_interval = 1.0 / max(infer_fps, 1e-6)  # 训练帧率对应的采样间隔(秒)
    # 初始化为"当前时间-一个间隔": 首帧立即采样, 且避免 0 与绝对时间戳比较导致 while 死循环
    last_sample_t = time.time() - sample_interval
    bboxes = []  # 避免首次未定义
    reconnect_tries = 0
    if profile:
        import collections
        prof = collections.defaultdict(list)

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            # 网络流断线 → 自动重连；本地文件/摄像头结束 → 退出
            if is_url:
                reconnect_tries += 1
                print(f"[流] 读取失败，第 {reconnect_tries} 次重连...")
                cap.release()
                if relay:
                    relay.terminate(); relay = None
                time.sleep(3)
                try:
                    cap, relay = _open_source(video_path, allow_relay=True)
                    reconnect_tries = 0
                    continue
                except RuntimeError:
                    if reconnect_tries >= 10:
                        print("重连失败超过 10 次，退出")
                        break
                    continue
            break

        if is_url and not play_only:
            # 只有当"上一帧处理耗时 > 一帧间隔"（处理跟不上流）时才丢弃积压帧保持画面最新；
            # 慢流/正常流下不做额外 grab，避免反复等待新帧而放大延迟
            if proc_ms_last > 1000.0 / fps_stream + 20.0:
                for _ in range(5):
                    if not cap.grab():
                        break
                ret2, frame = cap.retrieve()
                if not ret2:
                    continue

        t_read = time.time() - t0   # 读帧耗时（含丢帧处理）

        if play_only:
            # 只播放：不推理，仅显示 FPS 与帧号
            fps_buffer.append(1 / (time.time() - t0 + 0.001))
            fps = sum(fps_buffer) / len(fps_buffer)
            cv2.putText(frame, f"PLAY-ONLY | FPS:{fps:.0f} | Frame:{fidx}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if display_width:
                h, w = frame.shape[:2]
                scale = display_width / w
                frame = cv2.resize(frame, (display_width, int(h * scale)))
            cv2.imshow("Live Fall Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            fidx += 1
            continue

        # 时间采样：按训练帧率(infer_fps, 默认15)控制进入窗口的帧节奏, 与训练数据时间重采样一致
        # 源流 25/30fps 都能被均匀降到目标帧率(非整数倍也成立)
        t_infer_s = time.time()   # 推理段起点
        # 固定步进采样点, 避免每次重置到当前帧导致采样率累计漂移
        do_sample = (time.time() - last_sample_t) >= sample_interval
        if do_sample:
            last_sample_t += sample_interval            # 推进到下一个理想采样点
            while last_sample_t + sample_interval <= time.time():
                last_sample_t += sample_interval        # 落后太多则跳过旧采样点, 不追旧帧
            sample_cnt += 1

        if do_sample:
            # 检测：人体检测最耗时, 每 detect_every 次采样跑一次, 中间复用上一帧 bbox
            t_det_s = time.time()
            if sample_cnt % detect_every == 0 or not bboxes:
                if infer_width > 0:
                    # 推理分辨率: 检测前把图缩放到指定宽度(大分辨率流大幅提速), bbox 还原回原图坐标
                    fh, fw = frame.shape[:2]
                    det_frame = cv2.resize(frame, (infer_width, int(fh * infer_width / fw)))
                    boxes, scores = det_infer(det_frame)
                    sx = fw / det_frame.shape[1]
                    sy = fh / det_frame.shape[0]
                    bboxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                              for i, b in enumerate(boxes) if scores[i] > 0.3]
                else:
                    boxes, scores = det_infer(frame)   # 原图坐标 bbox (N,4), 分数降序
                    bboxes = [boxes[i] for i in range(len(scores)) if scores[i] > 0.3]
            else:
                # 复用检测框时向外扩 10%，容忍人体小幅移动
                _exp = []
                fw, fh = frame.shape[1], frame.shape[0]
                for b in bboxes:
                    x1, y1, x2, y2 = b
                    w, h = x2 - x1, y2 - y1
                    _exp.append([max(0, x1 - 0.1 * w), max(0, y1 - 0.1 * h),
                                 min(fw, x2 + 0.1 * w), min(fh, y2 + 0.1 * h)])
                bboxes = _exp
            t_det = time.time() - t_det_s

            kpts_raw = np.zeros((17, 3), dtype=np.float32)
            t_pose_s = time.time()
            if bboxes:
                res = pose_infer(frame, np.array(bboxes[:1]))
                if len(res) > 0:
                    r = res[0]
                    k = np.array(r.keypoints[0])
                    sc = np.array(r.keypoint_scores[0])
                    kpts_raw[:, :2] = k; kpts_raw[:, 2] = sc
                    kpts_raw[kpts_raw[:, 2] < 0.5] = 0.0
            t_pose = time.time() - t_pose_s
            # 归一化：髋居中 + 除身高（与训练一致）
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

            # 与训练一致：增强特征（速度=相邻帧差分, 只需上一帧+当前帧）
            norm_buffer.append(kpts_norm)
            if in_ch > 3:
                from features import enhance_keypoints
                prev = norm_buffer[-2] if len(norm_buffer) >= 2 else kpts_norm
                kpts_e = enhance_keypoints(np.stack([prev, kpts_norm]))[-1]  # (17, 7)
            else:
                kpts_e = kpts_norm
            kpt_buffer.append(kpts_e)
            kpt_buffer_raw.append(kpts_raw)
        else:
            # 未到采样时刻：不提取、不推进窗口, 用上一帧关键点画图
            t_det = t_pose = 0.0
            kpts_raw = kpt_buffer_raw[-1] if kpt_buffer_raw else np.zeros((17, 3), dtype=np.float32)

        # 绘制骨架（原始坐标）
        from pose_vis import draw_skeleton
        draw_skeleton(frame, kpts_raw, 0.5)
        risk_score, alert_text, color = 0.0, "SAFE", (0, 255, 0)
        if len(kpt_buffer) == window:
            x = torch.FloatTensor(np.array(list(kpt_buffer))).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0]
                nc = logits.shape[1]
                if nc >= 3:
                    risk_score = probs[1:].sum().item()
                    alert = fsm.update(probs.cpu().numpy())   # 3类概率[P0,P1,P2] → 状态机分级
                else:
                    risk_score = probs[1].item()
                    alert = fsm.update(risk_score)
            alert_text = alert.value
            colors = {"SAFE": (0, 255, 0), "LOW_RISK": (0, 255, 255),
                      "MID_RISK": (0, 165, 255), "HIGH_RISK": (0, 100, 255),
                      "IMMEDIATE": (0, 0, 255)}
            color = colors.get(alert_text, (255, 255, 255))
        t_infer = time.time() - t_infer_s   # 推理段耗时（检测+姿态+特征+分类）

        # UI 叠加
        h, w = frame.shape[:2]
        # 风险条
        cv2.rectangle(frame, (0, h - 40), (int(risk_score * w), h - 25), color, -1)
        cv2.rectangle(frame, (0, h - 40), (w, h - 25), (255, 255, 255), 1)
        cv2.putText(frame, f"{alert_text} ({risk_score:.2f})", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # FPS
        fps_buffer.append(1 / (time.time() - t0 + 0.001))
        fps = sum(fps_buffer) / len(fps_buffer)
        cv2.putText(frame, f"FPS:{fps:.0f} | Frame:{fidx} | Persons:{len(bboxes)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 缩放显示
        if display_width:
            h, w = frame.shape[:2]
            scale = display_width / w
            frame = cv2.resize(frame, (display_width, int(h * scale)))

        cv2.imshow("Live Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        proc_ms_last = (time.time() - t0) * 1000.0
        if profile:
            prof["read"].append(t_read * 1000.0)
            prof["infer"].append(t_infer * 1000.0)
            prof["det"].append(t_det * 1000.0)
            prof["pose"].append(t_pose * 1000.0)
            prof["disp"].append((time.time() - t0 - t_read - t_infer) * 1000.0)
            if fidx % 30 == 0 and len(prof["read"]) >= 5:
                avg = lambda k: sum(prof[k]) / len(prof[k])
                r, i, d = avg("read"), avg("infer"), avg("disp")
                det_ = avg("det") if prof["det"] else 0.0
                pos_ = avg("pose") if prof["pose"] else 0.0
                fc_ = max(0.0, i - det_ - pos_)
                print(f"[profile] F:{fidx} | 读帧:{r:7.1f} | 检测:{det_:6.1f} | 姿态:{pos_:6.1f} "
                      f"| 特征+分类:{fc_:6.1f} | 显示:{d:6.1f} | 合计:{r+i+d:7.1f}ms | FPS:{1000/(r+i+d):.1f}", flush=True)
        fidx += 1

    cap.release()
    if relay:
        relay.terminate()
    cv2.destroyAllWindows()
    print(f"\n结束，共 {fidx} 帧")


def main():
    p = argparse.ArgumentParser(description="实时跌倒检测")
    p.add_argument("--checkpoint", default=None,
                   help="模型权重路径（--play_only 时可不传）")
    p.add_argument("--video", type=str, default=None,
                    help="视频文件路径或 RTMP/RTSP/HTTP 网络流地址（含 & 需加引号），不填则用摄像头")
    p.add_argument("--extract_every", type=int, default=1,
                   help="(兼容保留) 采样节奏现由 --infer_fps 时间采样控制")
    p.add_argument("--infer_fps", type=float, default=15.0,
                   help="训练数据统一后的帧率, 推理按此帧率时间采样进窗口 (默认15)")
    p.add_argument("--window", type=int, default=30,
                   help=f"窗口帧数, 应与训练一致 (默认30, @15fps≈2秒)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--display_width", type=int, default=960, help="显示宽度像素，0=原始大小")
    p.add_argument("--play_only", action="store_true",
                   help="只播放视频流，不加载模型不推理（用于验证流/转码链路）")
    p.add_argument("--detect_every", type=int, default=1,
                   help="人体检测间隔帧数，中间帧复用上一帧检测框（默认1=每帧检测，可调大提帧率）")
    p.add_argument("--infer_width", type=int, default=0,
                   help="推理检测分辨率宽度(0=原图); 2K/高清流建议设960或640, 大幅提速")
    p.add_argument("--relay_width", type=int, default=0,
                   help="强制ffmpeg转码缩放宽度(>0), 网络读帧慢时大幅降码率; 0=不强制")
    p.add_argument("--profile", action="store_true",
                   help="打印每帧 读帧/推理/显示 分段耗时，用于定位瓶颈")
    a = p.parse_args()
    if not a.play_only and not a.checkpoint:
        p.error("未提供 --checkpoint（仅 --play_only 模式可省略）")
    demo_live(a.checkpoint, a.video, a.extract_every, a.window, a.device, a.display_width,
              a.play_only, a.detect_every, a.profile, a.infer_fps, a.infer_width, a.relay_width)


if __name__ == "__main__":
    main()
