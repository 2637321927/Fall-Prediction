"""
训练脚本: TCN / ST-GCN 步态不稳 & 摔倒检测 (3分类)
用法: python train.py --model tcn --epochs 50
      python train.py --model stgcn --epochs 50
"""

import torch, torch.nn as nn, numpy as np, argparse
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from tcn_model import TCN
from stgcn_model import STGCN
from dataset import FallDetectionDataset


def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in tqdm(loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward(); opt.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, crit, device, num_classes=3, fp_penalty=0.5):
    model.eval()
    loss_sum, preds, labels = 0, [], []
    for x, y in tqdm(loader, desc="Eval"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    preds, labels = np.array(preds), np.array(labels)

    # 每类指标
    per_class = {}
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    acc = (preds == labels).mean()
    # 宏平均 F1（各类权重相等）
    macro_f1 = np.mean([per_class[c]["f1"] for c in range(num_classes)])
    # 重点指标：class 1（步态不稳）的 F1
    unstable_f1 = per_class[1]["f1"] if 1 in per_class else 0
    # 压误报 + 兼顾精确率: avgF12(不稳+摔倒) 减去 误报率惩罚
    f1_1 = per_class[1]["f1"] if 1 in per_class else 0
    f1_2 = per_class[2]["f1"] if 2 in per_class else 0
    avg_f12 = (f1_1 + f1_2) / 2
    tp12 = per_class[1]["tp"] + per_class[2]["tp"]
    fp12 = per_class[1]["fp"] + per_class[2]["fp"]
    fp_rate = fp12 / (tp12 + fp12) if (tp12 + fp12) else 0.0  # 预测正例中误报比例
    score = avg_f12 - fp_penalty * fp_rate

    return {"loss": loss_sum / len(loader), "acc": acc,
            "macro_f1": macro_f1, "unstable_f1": unstable_f1,
            "avg_f12": avg_f12, "fp_rate": fp_rate, "score": score,
            "per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stgcn", choices=["tcn", "stgcn"],
                        help="模型类型: tcn 或 stgcn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6,
                        help="TCN层数 / ST-GCN块数")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="2(safe/risk) 或 4(normal/risk/falling/fallen)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="微调：加载已有模型权重继续训练")
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help="微调时学习率缩放因子")
    parser.add_argument("--features", type=str, default="basic",
                        choices=["basic", "enhanced"],
                        help="basic=仅(x,y,conf) enhanced=+速度+角度+重心")
    parser.add_argument("--fp_penalty", type=float, default=0.5,
                        help="score = avgF12 - fp_penalty*fp_rate; 越大越惩罚误报")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据：优先用真实 .npy + 标签，否则生成模拟数据
    data_dir = Path("../data/keypoints")
    label_dir = Path("../data/labels")
    npy_files = sorted(data_dir.glob("*_keypoints.npy"))

    videos = []  # [(名称, 关键点, 标签)]

    for npy_f in npy_files:
        # 跳过 _s_keypoints（旧版标注视频残留）
        if "_s_keypoints" in npy_f.name:
            continue
        label_f = label_dir / npy_f.name.replace("_keypoints.npy", "_labels.npy")
        if label_f.exists():
            k = np.load(npy_f)
            l = np.load(label_f)
            # 截断到较短者（关键点可能比标签少1帧）
            n = min(len(k), len(l))
            if n > 0:
                videos.append((npy_f.name.replace("_keypoints.npy", ""), k[:n], l[:n]))
                print(f"加载: {npy_f.name} ({n}帧)")

    if videos:
        # 统计真实类别数和分布
        all_labels_flat = np.concatenate([v[2] for v in videos])
        num_classes = max(args.num_classes, int(all_labels_flat.max()) + 1)
        print(f"真实数据: {len(videos)} 个视频, {len(all_labels_flat)} 帧")
        print(f"类别数: {num_classes} | normal={sum(all_labels_flat==0)} "
              f"unstable={sum(all_labels_flat==1)} falling={sum(all_labels_flat==2)} "
              f"fallen={sum(all_labels_flat==3)}")
    else:
        print("未找到标注数据，使用模拟数据...")
        np.random.seed(42)
        n = 2000
        safe = np.random.randn(n, 17, 3).astype(np.float32) * 0.3
        safe[:, :, 2] = np.abs(safe[:, :, 2])
        risk = np.random.randn(n, 17, 3).astype(np.float32) * 0.3 + 1.0
        risk[:, :, 2] = np.abs(risk[:, :, 2])
        videos = [("mock", np.concatenate([safe, risk]), np.array([0] * n + [1] * n))]
        num_classes = 2

    # 增强特征（逐视频处理，保证每视频首帧速度为0，不跨视频串扰）
    if args.features == "enhanced":
        from features import enhance_keypoints
        videos = [(name, enhance_keypoints(k), l) for name, k, l in videos]
        print(f"增强特征: {videos[0][1].shape}")

    # 按视频切分训练/验证集（避免同视频窗口泄漏到验证集）
    vid_idx = np.arange(len(videos))
    rng = np.random.RandomState(42)
    rng.shuffle(vid_idx)
    tr_size = int(0.8 * len(videos))
    tr_videos = [videos[i] for i in vid_idx[:tr_size]]
    vl_videos = [videos[i] for i in vid_idx[tr_size:]]

    tr_ds = FallDetectionDataset([v[1] for v in tr_videos], [v[2] for v in tr_videos],
                                 args.window, args.stride)
    vl_ds = FallDetectionDataset([v[1] for v in vl_videos], [v[2] for v in vl_videos],
                                 args.window, args.stride)
    tr_ld = DataLoader(tr_ds, args.batch, shuffle=True)
    vl_ld = DataLoader(vl_ds, args.batch)
    print(f"视频切分: 训练 {len(tr_videos)} 段, 验证 {len(vl_videos)} 段")
    print(f"窗口样本: 训练 {len(tr_ds)}, 验证 {len(vl_ds)}")

    # 创建模型: 输入通道数从增强后数据自动推断(而非写死, 兼容增强特征通道数变化)
    in_ch = videos[0][1].shape[2]
    if args.model == "tcn":
        model = TCN(hidden=args.hidden, num_layers=args.blocks, num_classes=num_classes, in_channels=in_ch).to(device)
    else:
        model = STGCN(hidden=args.hidden, num_blocks=args.blocks, num_classes=num_classes, in_channels=in_ch).to(device)
    print(f"模型: {args.model.upper()} | {num_classes}类 | 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 微调：加载已有权重
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"预训练加载: {args.pretrained} (F1={ckpt.get('f1',0):.4f})")

    lr = args.lr * (args.lr_factor if args.pretrained else 1.0)

    # 类别权重：少数类（不稳/摔倒）更高权重，解决不平衡
    all_labels_flat = np.concatenate([v[2] for v in videos])
    class_counts = np.bincount(all_labels_flat, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    print(f"类别权重: {class_weights.round(2).tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_score = -1.0
    Path("../checkpoints").mkdir(exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_ld, opt, crit, device)
        m = evaluate(model, vl_ld, crit, device, num_classes, args.fp_penalty)
        sch.step()
        pc = m['per_class']
        print(f"E{ep:3d} | L:{tr_loss:.3f} A:{tr_acc:.3f} | "
              f"V-L:{m['loss']:.3f} A:{m['acc']:.3f} "
              f"macroF1:{m['macro_f1']:.3f} avgF12:{m['avg_f12']:.3f} fpRate:{m['fp_rate']:.3f} "
              f"score:{m['score']:.3f}")
        for c in range(num_classes):
            p = pc[c]
            print(f"       class{c}: P={p['precision']:.3f} R={p['recall']:.3f} "
                  f"F1={p['f1']:.3f} (tp={p['tp']} fp={p['fp']} fn={p['fn']})")
        # 保存标准: score = avgF12(不稳+摔倒) - λ*误报率 (压误报+兼顾精确率)
        if m['score'] > best_score:
            best_score = m['score']
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "f1": m['avg_f12'], "macro_f1": m['macro_f1'],
                        "score": best_score, "fp_rate": float(m['fp_rate']),
                        "model_type": args.model, "num_classes": num_classes,
                        "config": {"hidden": args.hidden, "num_blocks": args.blocks}},
                       "../checkpoints/best_model.pt")
            print(f"  [SAVE] score={best_score:.4f} avgF12={m['avg_f12']:.4f} fpRate={m['fp_rate']:.3f}")

    print(f"\nDone! Best score: {best_score:.4f}")


if __name__ == "__main__":
    main()
