"""
评估已有的 best_model.pt: 在验证集上重新评估, 输出完整指标(每类P/R/F1、acc、score等)

用法:
  python eval_model.py --checkpoint ../checkpoints/best_model.pt
  python eval_model.py --checkpoint xxx.pt --fp_penalty 0.5 --device cuda

注意: 数据切分与 train.py/search.py 一致(RandomState(42) 按视频 8:2),
      保证验证集与训练时相同, 评估结果可对比。
"""
import torch, torch.nn as nn, numpy as np, argparse
from pathlib import Path

from tcn_model import TCN
from stgcn_model import STGCN
from dataset import FallDetectionDataset
from train import evaluate


def infer_in_ch(state):
    """从权重反推输入通道数"""
    for k, v in state.items():
        if "input_proj" in k and v.dim() == 4:
            return v.shape[1]
    return 3


def load_val_loader(args, in_ch):
    """加载数据(与 train.py 相同切分), 返回验证集 loader 和类别数"""
    kpt_dir = Path(args.data_dir) / "keypoints"
    label_dir = Path(args.data_dir) / "labels"
    videos = []
    for npy_f in sorted(kpt_dir.glob("*_keypoints.npy")):
        if "_s_keypoints" in npy_f.name:
            continue
        label_f = label_dir / npy_f.name.replace("_keypoints.npy", "_labels.npy")
        if label_f.exists():
            k = np.load(npy_f); l = np.load(label_f)
            n = min(len(k), len(l))
            if n > 0:
                videos.append((npy_f.name.replace("_keypoints.npy", ""), k[:n], l[:n]))
    if not videos:
        raise SystemExit(f"未找到数据: {kpt_dir}")
    num_classes = int(max(np.concatenate([v[2] for v in videos])) + 1)

    if in_ch > 3:
        from features import enhance_keypoints
        videos = [(n, enhance_keypoints(k), l) for n, k, l in videos]

    vid_idx = np.arange(len(videos))
    rng = np.random.RandomState(42)
    rng.shuffle(vid_idx)
    tr_size = int(0.8 * len(videos))
    vl_videos = [videos[i] for i in vid_idx[tr_size:]]
    vl_ds = FallDetectionDataset([v[1] for v in vl_videos], [v[2] for v in vl_videos],
                                 args.window, args.stride)
    vl_ld = torch.utils.data.DataLoader(vl_ds, args.batch)
    print(f"验证集: {len(vl_videos)} 段视频, {len(vl_ds)} 个窗口, {num_classes} 类, in_ch={in_ch}")
    return vl_ld, num_classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="../checkpoints/best_model.pt")
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--fp_penalty", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_json", type=str, default="")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mtype = ckpt.get("model_type", "stgcn")
    nc = ckpt.get("num_classes", 2)
    cfg = ckpt.get("config", {})
    in_ch = infer_in_ch(ckpt["model"])

    vl_ld, num_classes = load_val_loader(args, in_ch)

    if mtype == "tcn":
        model = TCN(hidden=cfg.get("hidden", 128), num_layers=cfg.get("num_blocks", 4),
                    num_classes=nc, in_channels=in_ch)
    else:
        model = STGCN(hidden=cfg.get("hidden", 128), num_blocks=cfg.get("num_blocks", 6),
                      num_classes=nc, in_channels=in_ch)
    model.load_state_dict(ckpt["model"]); model.to(device).eval()

    crit = nn.CrossEntropyLoss()
    m = evaluate(model, vl_ld, crit, device, num_classes, args.fp_penalty)
    pc = m['per_class']

    print("\n=== 评估结果 ===")
    print(f"  acc     : {m['acc']:.4f}")
    print(f"  macroF1 : {m['macro_f1']:.4f}")
    print(f"  avgF12  : {m['avg_f12']:.4f}")
    print(f"  fp_rate : {m['fp_rate']:.3f}")
    print(f"  score   : {m['score']:.4f} (= avgF12 - {args.fp_penalty}*fp_rate)")
    for c in range(num_classes):
        p_ = pc[c]
        print(f"  class{c}: P={p_['precision']:.4f} R={p_['recall']:.4f} F1={p_['f1']:.4f} "
              f"(tp={p_['tp']} fp={p_['fp']} fn={p_['fn']})")

    if args.save_json:
        import json
        metrics = {
            "checkpoint": args.checkpoint, "in_ch": in_ch,
            "acc": float(m['acc']), "macro_f1": float(m['macro_f1']),
            "avg_f12": float(m['avg_f12']), "fp_rate": float(m['fp_rate']),
            "score": float(m['score']),
            "per_class": {str(c): {k: (int(v) if k in ('tp', 'fp', 'fn') else float(v))
                                   for k, v in pc[c].items()} for c in range(num_classes)},
        }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"指标已保存: {args.save_json}")


if __name__ == "__main__":
    main()
