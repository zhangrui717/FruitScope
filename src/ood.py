"""开集拒识：负类采样、OOD 打分、阈值曲线。

计划书 v2 §4.4（开集安全）+ §4.6（坏案例证据）。

两种用法：
1) 训练时把 other 加进来（11 类）：
   python -m src.ood build-other --src data/raw/vegetables --per-class 60
2) 训练后评估拒识能力（不改权重）：
   python -m src.ood evaluate --ckpt ckpt/E03_best.pth --ood-dir data/ood
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.eval import collect_logits, evaluate_loader, load_model_from_checkpoint
from src.transforms import build_eval_transform, load_image
from src.utils import iter_files, save_json, set_seed


# ---------------- 负类采样 ----------------

TARGET_FRUITS = ("apple", "banana", "grape", "grapes", "kiwi", "mango",
                 "orange", "pear", "pineapple", "pomegranate", "watermelon")


def build_other_class(src_dirs, out_dir: str | Path, per_class: int = 60,
                      seed: int = config.SEED, dry_run: bool = False) -> dict:
    """从"非目标"目录（番茄/柠檬/蔬菜/杂物）每类抽 per_class 张，复制成 data/processed/other。

    这些图只作负类，文件名统一加 other_ 前缀，避免与水果目录混淆。
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    src_dirs = [Path(p) for p in src_dirs if Path(p).exists()]
    report = {"sources": [str(p) for p in src_dirs], "picked": {}, "total": 0}
    if not src_dirs:
        print("[warn] 没有可用的负类来源目录")
        return report
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src in src_dirs:
        for sub in sorted(p for p in src.iterdir() if p.is_dir()):
            if sub.name.lower() in TARGET_FRUITS:      # 目标水果不能混进负类
                continue
            files = list(iter_files(sub))
            if not files:
                continue
            take = min(per_class, len(files))
            picked = [files[i] for i in rng.choice(len(files), size=take, replace=False)]
            for f in picked:
                if not dry_run:
                    shutil.copy2(f, out_dir / f"other_{sub.name}_{f.name}")
            report["picked"][sub.name] = take
            report["total"] += take
    print(f"other 类共采样 {report['total']} 张 -> {out_dir}")
    return report


# ---------------- OOD 数据 ----------------

class UnlabeledImageDataset(Dataset):
    """只给图、不给标签的目录数据集（OOD / 自拍集用）。"""

    def __init__(self, root: str | Path, transform=None):
        self.paths = list(iter_files(Path(root)))
        if not self.paths:
            raise FileNotFoundError(f"{root} 下没有图片")
        self.transform = transform or build_eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return self.transform(load_image(self.paths[index]))


def build_ood_loader(root: str | Path, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    return DataLoader(UnlabeledImageDataset(root), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


# ---------------- OOD 打分与阈值 ----------------

def max_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """MSP：最大 softmax 概率，越低越像"没见过"。"""
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).max(axis=1)


def energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy = T * logsumexp(logits / T)，越小越 OOD。两种打分都保留，按实测选。"""
    z = logits / max(temperature, 1e-6)
    m = z.max(axis=1, keepdims=True)
    return float(temperature) * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


def rejection_report(id_scores: np.ndarray, ood_scores: np.ndarray, threshold: float) -> dict:
    """给定阈值，报告 OOD 拒识率（越高越好）与 ID 误拒率（越低越好）。"""
    ood_reject = float((ood_scores < threshold).mean()) if len(ood_scores) else float("nan")
    id_false_reject = float((id_scores < threshold).mean()) if len(id_scores) else float("nan")
    return {"threshold": float(threshold), "ood_reject": ood_reject,
            "false_reject": id_false_reject,
            "n_id": int(len(id_scores)), "n_ood": int(len(ood_scores))}


def sweep_threshold(id_scores: np.ndarray, ood_scores: np.ndarray,
                    target_false_reject: float = 0.05, n_grid: int = 400) -> dict:
    """在"ID 误拒率 <= target"的约束下，选 OOD 拒识率最高的阈值。"""
    empty = {"threshold": float("nan"), "ood_reject": float("nan"),
             "false_reject": float("nan"), "n_id": int(len(id_scores)),
             "n_ood": int(len(ood_scores))}
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return empty
    lo = float(min(id_scores.min(), ood_scores.min()))
    hi = float(max(id_scores.max(), ood_scores.max()))
    best = None
    for t in np.linspace(lo, hi, n_grid):
        rep = rejection_report(id_scores, ood_scores, t)
        if rep["false_reject"] > target_false_reject:
            continue
        if best is None or rep["ood_reject"] > best["ood_reject"] + 1e-12:
            best = rep          # 同分时保留更低的阈值（误拒更少，更保守）
    return best or empty


def ood_argmax_report(ood_logits: np.ndarray, classes: Sequence[str]) -> dict:
    """有 other 类时，上线真正生效的是"OOD 落到 other 的比例"，而不是阈值拒识。

    10 类模型只能靠 MSP 阈值拒识；11 类模型应该让 other 吸收陌生输入。
    这里把 OOD 集上的预测分布摊开，便于判断 other 是否真的学会了"不是水果"。
    """
    idx = np.argmax(ood_logits, axis=1)
    counts = np.bincount(idx, minlength=len(classes))
    n = int(len(idx))
    other_idx = classes.index(config.OTHER_CLASS) if config.OTHER_CLASS in classes else -1
    top = sorted(((classes[i], int(c)) for i, c in enumerate(counts) if c), key=lambda kv: -kv[1])
    as_other = (counts[other_idx] / n) if other_idx >= 0 and n else None
    return {
        "n_ood": n,
        "has_other_class": other_idx >= 0,
        "ood_as_other": as_other,
        "ood_as_fruit": (1.0 - as_other) if as_other is not None else 1.0,
        "top_pred": [{"label": k, "count": v, "share": v / n} for k, v in top[:8]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="开集负类构造 / OOD 评估")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-other", help="从非目标目录采样成 other 类")
    b.add_argument("--src", nargs="+", required=True)
    b.add_argument("--out", default=str(config.PROCESSED_DIR / config.OTHER_CLASS))
    b.add_argument("--per-class", type=int, default=60)
    b.add_argument("--dry-run", action="store_true")

    e = sub.add_parser("evaluate", help="评估拒识率 / 误拒率")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default=str(config.SPLIT_FILE))
    e.add_argument("--ood-dir", default=str(config.OOD_DIR))
    e.add_argument("--score", default="msp", choices=["msp", "energy"])
    e.add_argument("--target-frr", type=float, default=0.05,
                   help="允许的正常样本误拒率上限")
    e.add_argument("--save-json", default=None)
    args = ap.parse_args()

    if args.cmd == "build-other":
        report = build_other_class(args.src, args.out, args.per_class, dry_run=args.dry_run)
        save_json(config.DOCS_DIR / "other_class_report.json", report)
        return

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model_from_checkpoint(args.ckpt, device)
    temperature = float(ckpt.get("temperature") or 1.0)

    from src.dataset import build_all_loaders
    loaders, classes, _ = build_all_loaders(args.split, 32, 0)
    id_logits, _ = collect_logits(model, loaders["val"], device)      # 阈值在 val 上定
    test_metrics = evaluate_loader(model, loaders["test"], classes, device, temperature)
    print(f"ID(test): acc={test_metrics['acc']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")

    ood_loader = build_ood_loader(args.ood_dir, batch_size=32, num_workers=0)
    ood_logits, _ = collect_logits(model, ood_loader, device)

    scorer = max_softmax if args.score == "msp" else energy
    id_scores = scorer(id_logits, temperature)
    ood_scores = scorer(ood_logits, temperature)
    best = sweep_threshold(id_scores, ood_scores, args.target_frr)
    print(f"score={args.score} 建议阈值={best['threshold']:.4f} "
          f"OOD拒识={best['ood_reject']:.4f} ID误拒={best['false_reject']:.4f} "
          f"(n_id={best['n_id']}, n_ood={best['n_ood']})")

    payload = {"score": args.score, "temperature": temperature, "val": best,
               "test": {k: test_metrics[k] for k in ("acc", "acc_ci95", "macro_f1", "top3", "ece")}}
    if args.save_json:
        save_json(Path(args.save_json), payload)
        print(f"已保存 {args.save_json}")


if __name__ == "__main__":
    main()
