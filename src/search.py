"""
网格搜索 + 动态调参: 遍历超参组合, 按 score(avgF12 - λ*误报率) 选最优, 再用最优超参重训

用法:
  python search.py --grid small --epochs 25 --final_epochs 100 --features enhanced
  python search.py --grid large --epochs 25 --final_epochs 100 --fp_penalty 0.8

指标说明:
  score = avgF12(class1+class2 平均F1) - fp_penalty * fp_rate
  兼顾"不稳/摔倒"精确率, 同时惩罚误报
"""
import torch, torch.nn as nn, numpy as np, argparse, itertools, time
from pathlib import Path
from tqdm import tqdm

from tcn_model import TCN
from stgcn_model import STGCN
from dataset import FallDetectionDataset
from train import train_epoch, evaluate


def load_data(args):
    """加载数据 + 增强 + 按视频切分 (与 train.py 一致)"""
    data_dir = Path(args.data_dir) / "keypoints"
    label_dir = Path(args.data_dir) / "labels"
    npy_files = sorted(data_dir.glob("*_keypoints.npy"))

    videos = []
    for npy_f in npy_files:
        if "_s_keypoints" in npy_f.name:
            continue
        label_f = label_dir / npy_f.name.replace("_keypoints.npy", "_labels.npy")
        if label_f.exists():
            k = np.load(npy_f)
            l = np.load(label_f)
            n = min(len(k), len(l))
            if n > 0:
                videos.append((npy_f.name.replace("_keypoints.npy", ""), k[:n], l[:n]))

    if not videos:
        raise SystemExit(f"未找到数据: {data_dir}/*_keypoints.npy")

    all_labels_flat = np.concatenate([v[2] for v in videos])
    num_classes = max(args.num_classes, int(all_labels_flat.max()) + 1)
    print(f"真实数据: {len(videos)} 个视频, {len(all_labels_flat)} 帧, {num_classes} 类")

    # 增强特征
    if args.features == "enhanced":
        from features import enhance_keypoints
        videos = [(name, enhance_keypoints(k), l) for name, k, l in videos]
        print(f"增强特征: {videos[0][1].shape}")

    # 按视频切分(避免同视频泄漏)
    vid_idx = np.arange(len(videos))
    rng = np.random.RandomState(42)
    rng.shuffle(vid_idx)
    tr_size = int(0.8 * len(videos))
    tr_videos = [videos[i] for i in vid_idx[:tr_size]]
    vl_videos = [videos[i] for i in vid_idx[tr_size:]]

    tr_ds = FallDetectionDataset([v[1] for v in tr_videos], [v[2] for v in tr_videos], args.window, args.stride)
    vl_ds = FallDetectionDataset([v[1] for v in vl_videos], [v[2] for v in vl_videos], args.window, args.stride)
    tr_ld = torch.utils.data.DataLoader(tr_ds, args.batch, shuffle=True)
    vl_ld = torch.utils.data.DataLoader(vl_ds, args.batch)

    tr_counts = np.bincount(np.concatenate([v[2] for v in tr_videos]), minlength=num_classes).astype(np.float32)
    print(f"切分: 训练{len(tr_videos)}段({len(tr_ds)}窗口) 验证{len(vl_videos)}段({len(vl_ds)}窗口)")
    return tr_ld, vl_ld, num_classes, videos[0][1].shape[2], tr_counts


def build_model(mtype, in_ch, num_classes, hidden, blocks, device):
    if mtype == "tcn":
        model = TCN(hidden=hidden, num_layers=blocks, num_classes=num_classes, in_channels=in_ch)
    else:
        model = STGCN(hidden=hidden, num_blocks=blocks, num_classes=num_classes, in_channels=in_ch)
    return model.to(device)


def train_combo(model, tr_ld, vl_ld, num_classes, tr_counts, lr, epochs, fp_penalty, device):
    """训练一组超参, 返回验证集最优指标(按 score)"""
    class_weights = 1.0 / (tr_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best = None
    for ep in range(1, epochs + 1):
        train_epoch(model, tr_ld, opt, crit, device)
        m = evaluate(model, vl_ld, crit, device, num_classes, fp_penalty)
        sch.step()
        if best is None or m['score'] > best['score']:
            best = m
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=str, default="small", choices=["small", "large"])
    p.add_argument("--epochs", type=int, default=25, help="每组筛选轮数")
    p.add_argument("--final_epochs", type=int, default=100, help="最优组合最终重训轮数")
    p.add_argument("--model", type=str, default="stgcn", choices=["tcn", "stgcn"])
    p.add_argument("--features", type=str, default="enhanced", choices=["basic", "enhanced"])
    p.add_argument("--fp_penalty", type=float, default=0.5)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save", type=str, default="../checkpoints/best_model.pt")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tr_ld, vl_ld, num_classes, in_ch, tr_counts = load_data(args)

    if args.grid == "small":
        grid = {"hidden": [128, 256], "blocks": [4, 6], "lr": [1e-3, 5e-4]}
    else:
        grid = {"hidden": [64, 128, 256], "blocks": [4, 6, 8], "lr": [1e-3, 5e-4, 1e-4]}
    combos = list(itertools.product(grid["hidden"], grid["blocks"], grid["lr"]))
    print(f"\n网格: {len(combos)} 组 | 每组 {args.epochs} epoch | fp_penalty={args.fp_penalty}\n")

    results = []
    for idx, (hidden, blocks, lr) in enumerate(combos, 1):
        t0 = time.time()
        model = build_model(args.model, in_ch, num_classes, hidden, blocks, device)
        best = train_combo(model, tr_ld, vl_ld, num_classes, tr_counts, lr, args.epochs, args.fp_penalty, device)
        pc = best['per_class']
        fp_total = pc[1]['fp'] + pc[2]['fp']
        print(f"[{idx}/{len(combos)}] hidden={hidden} blocks={blocks} lr={lr} "
              f"→ score={best['score']:.4f} avgF12={best['avg_f12']:.4f} "
              f"fpRate={best['fp_rate']:.3f} fp1+fp2={fp_total} acc={best['acc']:.3f} "
              f"({time.time()-t0:.0f}s)")
        results.append(((hidden, blocks, lr), best))

    results.sort(key=lambda r: -r[1]['score'])
    print("\n=== 排序 (按 score) ===")
    for i, (combo, m) in enumerate(results, 1):
        pc = m['per_class']
        print(f"  #{i} hidden={combo[0]:<4} blocks={combo[1]} lr={combo[2]:<6} "
              f"score={m['score']:.4f} avgF12={m['avg_f12']:.4f} "
              f"fpRate={m['fp_rate']:.3f} fp1+fp2={pc[1]['fp']+pc[2]['fp']}")

    best_combo, best_m = results[0]
    print(f"\n最优组合: hidden={best_combo[0]} blocks={best_combo[1]} lr={best_combo[2]}")
    print(f"用最优超参重训 {args.final_epochs} 轮...")

    model = build_model(args.model, in_ch, num_classes, best_combo[0], best_combo[1], device)
    class_weights = 1.0 / (tr_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=best_combo[2], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.final_epochs)
    best_final = None
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.final_epochs + 1):
        train_epoch(model, tr_ld, opt, crit, device)
        m = evaluate(model, vl_ld, crit, device, num_classes, args.fp_penalty)
        sch.step()
        if best_final is None or m['score'] > best_final['score']:
            best_final = m
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "f1": m['avg_f12'], "macro_f1": m['macro_f1'],
                        "score": best_final['score'], "fp_rate": float(m['fp_rate']),
                        "model_type": args.model, "num_classes": num_classes,
                        "config": {"hidden": best_combo[0], "num_blocks": best_combo[1]}},
                       args.save)
    print(f"\n最终 best_model 保存: {args.save}")

    # 完整评估分数
    pc = best_final['per_class']
    print("=== 最终最佳模型完整评估 ===")
    print(f"  acc     : {best_final['acc']:.4f}")
    print(f"  macroF1 : {best_final['macro_f1']:.4f}")
    print(f"  avgF12  : {best_final['avg_f12']:.4f}  (class1+class2 平均F1)")
    print(f"  fp_rate : {best_final['fp_rate']:.3f}  (预测正例中误报比例)")
    print(f"  score   : {best_final['score']:.4f}  (= avgF12 - {args.fp_penalty}*fp_rate)")
    for c in range(num_classes):
        p = pc[c]
        print(f"  class{c}: P={p['precision']:.4f} R={p['recall']:.4f} F1={p['f1']:.4f} "
              f"(tp={p['tp']} fp={p['fp']} fn={p['fn']})")

    # 指标保存到 json
    try:
        import json
        metrics = {
            "best_combo": {"hidden": best_combo[0], "blocks": best_combo[1], "lr": best_combo[2]},
            "acc": float(best_final['acc']),
            "macro_f1": float(best_final['macro_f1']),
            "avg_f12": float(best_final['avg_f12']),
            "fp_rate": float(best_final['fp_rate']),
            "score": float(best_final['score']),
            "per_class": {str(c): {k: (int(v) if k in ('tp', 'fp', 'fn') else float(v))
                                   for k, v in pc[c].items()} for c in range(num_classes)},
        }
        mpath = str(Path(args.save).with_suffix(".metrics.json"))
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"指标已保存: {mpath}")
    except Exception as e:
        print(f"指标保存失败: {e}")


if __name__ == "__main__":
    main()
