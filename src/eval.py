"""评估：收集 logits -> 指标（acc + 95% CI / macro-F1 / Top-3 / 混淆矩阵 / 逐类报告）。

计划书 v2 §4.4：小测试集必须带置信区间，还要报 Top-3 与校准后的 ECE。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import config
from src.dataset import build_all_loaders
from src.model import build_from_checkpoint
from src.utils import (confusion_matrix, expected_calibration_error, macro_f1,
                       per_class_report, save_json, set_seed, softmax_np, topk_correct, wilson_ci)


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader, device: torch.device):
    """跑完一个 loader，返回 (logits[N,C], targets[N])。

    兼容两种 loader：带标签的 (images, targets) 与 OOD 的无标签 images。
    """
    model.eval()
    logits_all, targets_all = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, targets = batch
        else:
            images, targets = batch, None
        images = images.to(device, non_blocking=True)
        out = model(images)
        logits_all.append(out.detach().float().cpu())
        if targets is not None:
            targets_all.append(targets.detach().cpu())
    if not logits_all:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    targets = (torch.cat(targets_all).numpy().astype(np.int64) if targets_all
               else np.zeros((0,), dtype=np.int64))
    return torch.cat(logits_all).numpy(), targets


def summarize(logits: np.ndarray, targets: np.ndarray, classes, temperature: float = 1.0) -> dict:
    """一次性算齐所有指标，任何脚本都调它，避免"两套口径"。"""
    n = int(targets.shape[0])
    if n == 0 or logits.size == 0:
        return {"n": 0, "acc": float("nan"), "acc_ci95": [float("nan")] * 2,
                "macro_f1": float("nan"), "top3": float("nan"), "ece": float("nan"),
                "per_class": [], "confusion": []}
    probs = softmax_np(logits / max(temperature, 1e-6))
    pred = logits.argmax(axis=1)
    correct = int((pred == targets).sum())
    lo, hi = wilson_ci(correct, n)
    return {
        "n": n,
        "acc": correct / n,
        "acc_ci95": [round(lo, 4), round(hi, 4)],
        "macro_f1": macro_f1(targets, pred, len(classes)),
        "top3": topk_correct(logits, targets, k=3) / n,
        "ece": expected_calibration_error(probs, targets),
        "per_class": per_class_report(targets, pred, list(classes)),
        "confusion": confusion_matrix(targets, pred, len(classes)).tolist(),
    }


def evaluate_loader(model, loader, classes, device, temperature: float = 1.0) -> dict:
    logits, targets = collect_logits(model, loader, device)
    return summarize(logits, targets, classes, temperature)


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device):
    """重建结构并载入权重；返回 (model, ckpt)。"""
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    model = build_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def save_confusion_png(cm: np.ndarray, classes, out_path: str | Path,
                       title: str = "confusion matrix") -> None:
    """画行归一化混淆矩阵；没装 matplotlib 就退化成 txt，不阻断流程。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        np.savetxt(out_path.with_suffix(".txt"), np.asarray(cm), fmt="%d")
        print(f"[warn] 未安装 matplotlib，混淆矩阵改存为 {out_path.with_suffix('.txt')}")
        return

    cm = np.asarray(cm, dtype=np.float64)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(max(6.0, len(classes) * 0.6), max(5.0, len(classes) * 0.55)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="在指定划分上评估 checkpoint")
    ap.add_argument("--ckpt", required=True, help="例如 ckpt/E03_best.pth")
    ap.add_argument("--split", default=str(config.SPLIT_FILE))
    ap.add_argument("--which", default="test", choices=["train", "val", "test"])
    ap.add_argument("--temperature", type=float, default=None,
                    help="默认用 checkpoint 里标定好的温度")
    ap.add_argument("--batch-size", type=int, default=config.DEFAULTS["batch_size"])
    ap.add_argument("--num-workers", type=int, default=config.DEFAULTS["num_workers"])
    ap.add_argument("--confusion-png", default=None)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model_from_checkpoint(args.ckpt, device)
    temperature = (args.temperature if args.temperature is not None
                   else float(ckpt.get("temperature") or 1.0))

    loaders, classes, sizes = build_all_loaders(args.split, args.batch_size, args.num_workers)
    metrics = evaluate_loader(model, loaders[args.which], classes, device, temperature)

    print(f"== {args.which} (n={metrics['n']}) ==")
    print(f"acc      : {metrics['acc']:.4f}  95%CI {metrics['acc_ci95']}")
    print(f"macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"top-3    : {metrics['top3']:.4f}")
    print(f"ECE      : {metrics['ece']:.4f}   (T={temperature:.3f})")
    print("per-class:")
    for row in metrics["per_class"]:
        print(f"  {row['class']:<12} P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f} n={row['support']}")

    if args.confusion_png:
        save_confusion_png(np.asarray(metrics["confusion"]), classes, args.confusion_png,
                           title=f"{Path(args.ckpt).stem} / {args.which}")
        print(f"混淆矩阵: {args.confusion_png}")
    if args.save_json:
        save_json(Path(args.save_json), metrics)
        print(f"指标 JSON: {args.save_json}")


if __name__ == "__main__":
    main()
