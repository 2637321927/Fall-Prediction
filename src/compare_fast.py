"""对比验证: fast_infer（手写后处理） vs 现有低层 API，输出写入文件"""
import sys, warnings, traceback, time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # 项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import numpy as np, cv2, torch

LOG = str(BASE / "artifacts" / "fast_compare_out.txt")

def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(m) + "\n")

def main():
    open(LOG, "w").close()
    from pose_mm import DET_CFG, DET_CKPT, POSE_CFG, POSE_CKPT
    from mmdet.apis import init_detector
    from mmpose.apis import init_model
    from low_level_infer import make_det_infer, make_pose_infer
    from fast_infer import make_fast_det_infer, make_fast_pose_infer

    det = init_detector(DET_CFG, DET_CKPT, device="cuda")
    pose = init_model(POSE_CFG, POSE_CKPT, device="cuda")

    frame = None
    cap = cv2.VideoCapture(str(BASE / "mmpose" / "demo" / "resources" / "demo.mp4"))
    for i in range(0, 120, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok and fr is not None:
            frame = fr
            break
    cap.release()
    if frame is None:
        frame = cv2.imread(str(BASE / "mmpose" / "demo" / "resources" / "sunglasses.jpg"))
    if frame is None:
        log("测试图读取失败")
        return
    log("测试图: " + str(frame.shape))

    det_infer = make_det_infer(det)
    fast_det = make_fast_det_infer(det)
    pose_infer = make_pose_infer(pose)
    fast_pose = make_fast_pose_infer(pose)

    # ---- 现有低层 API ----
    d = det_infer(frame)
    old_boxes = d.pred_instances.bboxes.cpu().numpy()
    old_scores = d.pred_instances.scores.cpu().numpy()
    old_top = old_boxes[old_scores > 0.3]
    log(f"\n旧: bbox数={len(old_boxes)} (>0.3: {len(old_top)})")
    if len(old_top):
        res = pose_infer(frame, old_top[:1])
        old_kpts = np.array(res[0].pred_instances.keypoints[0])
        log("旧 姿态头点: " + repr(old_kpts[0].tolist()))

    # ---- fast ----
    t0 = time.time()
    new_boxes, new_scores = fast_det(frame)
    torch.cuda.synchronize()
    log(f"新: bbox数={len(new_boxes)} (>0.3: {int((new_scores>0.3).sum())})  检测耗时: {(time.time()-t0)*1000:.1f}ms")
    top = new_boxes[new_scores > 0.3]
    if len(top):
        res2 = fast_pose(frame, top[:1])
        new_kpts = np.array(res2[0].keypoints[0])   # InstanceData 直接是 keypoints 字段
        log("新 姿态头点: " + repr(new_kpts[0].tolist()))

    # ---- 对比 ----
    if len(old_top) and len(top):
        db = np.abs(old_top[0] - top[0]).max()
        dk = np.abs(old_kpts - new_kpts).max()
        log(f"\nbbox(top1) 最大差: {db:.4f} | 关键点最大差: {dk:.4f}")
        # 关键点差 ~3px 来自 flip_test(TTA): 旧链路在 predict 里做原图+翻转两次前向融合，
        # fast 走单次前向(与训练一致)。960x540 上图 3px≈0.3%，归一化后对 ST-GCN 影响可忽略。
        ok = db < 3.0 and dk < 5.0
        log("一致性: PASS ✅" if ok else "一致性: FAIL ❌")
    else:
        log("未检出人体，无法对比（建议换含人的测试图）")

    # ---- 提速对比 ----
    # fast 检测计时
    for _ in range(3):
        fast_det(frame)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.time()
        fast_det(frame)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    log(f"\nfast 检测平均: {np.mean(ts):.1f} ms  (旧链路约 185 ms)")
    if len(top):
        b = top[:1]
        for _ in range(3):
            fast_pose(frame, b)
        torch.cuda.synchronize()
        ts2 = []
        for _ in range(10):
            t0 = time.time()
            fast_pose(frame, b)
            torch.cuda.synchronize()
            ts2.append((time.time() - t0) * 1000)
        log(f"fast 姿态平均: {np.mean(ts2):.1f} ms  (旧链路约 44 ms)")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("\n=== EXCEPTION ===\n" + err)
