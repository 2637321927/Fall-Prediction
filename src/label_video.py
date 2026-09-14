"""
视频标注工具: 支持标记多段跌倒事件
用法: python label_video.py --video fall1.mp4

操作:
  空格 = 播放/暂停       ←→ = 逐帧
  +    = 新增一段跌倒      D = 删除当前事件
  1-9  = 切换到第N段事件
  R    = 标 risk_start     F = 标 fall_start
  I    = 标 impact         S = 标 stable_fallen
  N    = 清空所有事件(正常视频)
  Q    = 保存并退出

输出: *_labels.npy (0=normal, 1=falling, 2=fallen) 3分类
       *_info.json (标注详情)
"""

import cv2, numpy as np, json, argparse
from pathlib import Path


def label_video(video_path, save_dir="../data/labels"):
    vp = Path(video_path)
    if not vp.exists(): print(f"视频不存在: {vp}"); return

    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dw = min(w, 1280); dh = int(dw / w * h)
    cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Tool", dw, dh)

    events = []   # [{"risk":int, "fall":int, "impact":int, "stable":int}, ...]
    cur = 0
    fidx, playing = 0, True

    print(f"\n{'='*55}\n {vp.name} | {total}帧 | {fps:.1f}fps\n{'='*55}")
    print(" [+]新增 [D]删除 [1-9]切换 [R/F/I/S]标记 [Space]播放 [Q]保存")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret: break

        label, color, status = _get_label(fidx, events)
        frame = _draw_ui(frame, fidx, total, fps, events, cur, label, color, status, w, h)
        cv2.imshow("Label Tool", frame)
        key = cv2.waitKey(0 if not playing else 25) & 0xFF

        if key == ord('q'): break
        elif key == ord(' '): playing = not playing
        elif key == 81: fidx = max(0, fidx - 1); playing = False
        elif key == 83: fidx = min(total - 1, fidx + 1); playing = False
        elif key in (ord('+'), ord('=')):
            events.append(dict(risk=None, fall=None, impact=None, stable=None))
            cur = len(events) - 1
            print(f"  + 事件 #{cur+1}")
        elif key in (ord('r'), ord('R')) and events:
            events[cur]["risk"] = fidx; print(f"  事件{cur+1} R={fidx}")
        elif key in (ord('f'), ord('F')) and events:
            events[cur]["fall"] = fidx; print(f"  事件{cur+1} F={fidx}")
        elif key in (ord('i'), ord('I')) and events:
            events[cur]["impact"] = fidx; print(f"  事件{cur+1} I={fidx}")
        elif key in (ord('s'), ord('S')) and events:
            events[cur]["stable"] = fidx; print(f"  事件{cur+1} S={fidx}")
        elif key in (ord('d'), ord('D')) and events:
            del events[cur]; cur = max(0, len(events) - 1)
            print(f"  - 删除, 剩余 {len(events)} 段")
        elif key in (ord('n'), ord('N')):
            events.clear(); cur = 0; print("  清空 → 正常视频")
        elif ord('1') <= key <= ord('9'):
            n = key - ord('1')
            if n < len(events): cur = n; print(f"  切换到 #{n+1}")
        elif playing:
            fidx = min(total - 1, fidx + 1)
        if fidx >= total - 1: playing = False

    cap.release(); cv2.destroyAllWindows()

    labels = _gen_labels(total, events)
    d = Path(save_dir); d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{vp.stem}_labels.npy", labels)
    json.dump({"video": vp.name, "total": total, "fps": fps, "events": events,
               "counts": dict(zip([0,1,2], np.bincount(labels, minlength=3).tolist()))},
              open(d / f"{vp.stem}_info.json", "w"), indent=2, ensure_ascii=False)
    _summary(events, labels)


def _get_label(fidx, events):
    for ev in reversed(events):
        s, r = ev.get("stable"), ev.get("risk")
        if s is not None and fidx >= s: return 2, (0, 0, 255), "FALLEN"
        if r is not None and fidx >= r: return 1, (0, 165, 255), "FALLING"
    return 0, (0, 255, 0), "NORMAL"


def _draw_ui(frame, fidx, total, fps, events, cur, label, color, status, w, h):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 92), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.75, ov, 0.25, 0)

    cv2.putText(frame, f"Frame:{fidx}/{total} ({fidx/fps:.1f}s) | Events:{len(events)}",
                (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, status, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 事件列表（右上）
    cols = [(0, 255, 255), (0, 165, 255), (0, 100, 255), (0, 0, 255)]
    for ei, ev in enumerate(events):
        mrk = ">" if ei == cur else " "
        parts = [f"{mrk}#{ei+1}:"]
        for nm, cl in zip(["risk","fall","impact","stable"], cols):
            v = ev.get(nm)
            parts.append(f"[{nm[0].upper()}{v if v is not None else '?'}]")
        cv2.putText(frame, "".join(parts), (w - 400, 22 + ei * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 时间线
    for ei, ev in enumerate(events):
        y = 6 + ei * 10
        for k, cl in zip(["risk","fall","impact","stable"], cols):
            v = ev.get(k)
            if v is not None:
                cv2.line(frame, (int(v/total*w), y), (int(v/total*w), y+7), cl, 1)

    # 底栏
    cv2.rectangle(frame, (0, h-26), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "[+]New [D]el [N]ormal [1-9]Switch [R/F/I/S]Mark [Space]Play [Q]uit",
                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return frame


def _gen_labels(total, events):
    """0=normal, 1=falling(risk~stable), 2=fallen"""
    labels = np.zeros(total, dtype=np.int64)
    for ev in events:
        r, s = ev.get("risk"), ev.get("stable")
        if r: labels[r:] = np.maximum(labels[r:], 1)
        if s: labels[s:] = np.maximum(labels[s:], 2)
    return labels


def _summary(events, labels):
    print(f"\n{'='*55}\n完成! {len(events)}段跌倒")
    for ei, ev in enumerate(events):
        print(f"  #{ei+1}: R={ev['risk']} F={ev.get('fall','?')} I={ev.get('impact','?')} S={ev['stable']}")
    cnt = np.bincount(labels, minlength=3)
    print(f"  分布: normal={cnt[0]} falling={cnt[1]} fallen={cnt[2]}")
    print("="*55)


def main():
    p = argparse.ArgumentParser(description="跌倒标注 (多段事件)")
    p.add_argument("--video", required=True)
    p.add_argument("--save_dir", type=str, default="../data/labels")
    a = p.parse_args()
    label_video(a.video, a.save_dir)


if __name__ == "__main__":
    main()
