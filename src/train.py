"""训练入口：冻结训头 -> 条件解冻 -> 评估 -> 存 checkpoint -> 追加 runs.csv。

计划书 v2 §4.3 / §4.5 / §4.7 的落地实现。

用法（单卡 3090 上 10 类约 1 分钟/epoch 以内）：
    python -m src.train --tag E03_mbv3l_linear --epochs 30
    python -m src.train --tag E04_mlp --head mlp --epochs 30
    python -m src.train --tag E05_unfreeze3 --unfreeze-last 3 --unfreeze-epochs 15
    nohup python -m src.train --tag E03 > logs/E03.out 2>&1 &
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import build_all_loaders
from src.eval import evaluate_loader, save_confusion_png
from src.model import (build_model, build_optimizer, count_params, freeze_backbone, head_config,
                       make_scheduler, set_frozen_bn_eval, unfreeze_last_blocks)
from src.utils import (Tee, append_run_csv, fmt_metrics, get_device, git_commit, load_json,
                       now_str, save_json, set_seed)

AUG_TAG = "crop0.6/flip/affine15/jit0.2/erase0.25"


def env_info() -> dict:
    """写进 checkpoint 的环境信息：换机器复现时先看这里。"""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torchvision": __import__("torchvision").__version__,
    }
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def parse_args() -> argparse.Namespace:
    d = config.DEFAULTS
    ap = argparse.ArgumentParser(description="FruitScope 迁移学习训练")
    ap.add_argument("--tag", default="E00_baseline", help="实验编号，决定 ckpt/log 文件名")
    ap.add_argument("--split", default=str(config.SPLIT_FILE))
    ap.add_argument("--arch", default=d["arch"], choices=["mobilenet_v3_large", "mobilenet_v3_small"])
    ap.add_argument("--head", default=d["head"], choices=["linear", "mlp", "official"])
    ap.add_argument("--hidden", type=int, default=d["hidden"])
    ap.add_argument("--dropout", type=float, default=d["dropout"])
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--unfreeze-epochs", type=int, default=d["unfreeze_epochs"])
    ap.add_argument("--batch-size", type=int, default=d["batch_size"])
    ap.add_argument("--lr-head", type=float, default=d["lr_head"])
    ap.add_argument("--lr-bb", type=float, default=d["lr_bb"])
    ap.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    ap.add_argument("--label-smoothing", type=float, default=d["label_smoothing"])
    ap.add_argument("--patience", type=int, default=d["patience"])
    ap.add_argument("--num-workers", type=int, default=d["num_workers"])
    ap.add_argument("--unfreeze-last", type=int, default=d["unfreeze_last"],
                    help="0=全程冻结；3=解冻 features[-3:]")
    ap.add_argument("--weighted-sampler", action="store_true", help="类别失衡时开")
    ap.add_argument("--no-pretrained", action="store_true", help="冒烟测试用（不下载权重）")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--ckpt-dir", default=str(config.CKPT_DIR), help="冒烟测试可指向临时目录")
    ap.add_argument("--runs-csv", default=str(config.RUNS_CSV), help="实验记录表路径")
    ap.add_argument("--curves-dir", default=str(config.DOCS_DIR / "curves"),
                    help="混淆矩阵图片目录")
    ap.add_argument("--note", default="")
    return ap.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """一个 epoch：train 时反传，返回 (loss, logits, targets)。"""
    model.train(train)
    total_loss, n_seen = 0.0, 0
    logits_all, targets_all = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        n_seen += targets.size(0)
        logits_all.append(logits.detach().float().cpu())
        targets_all.append(targets.detach().cpu())
    return (total_loss / max(n_seen, 1), torch.cat(logits_all).numpy(), torch.cat(targets_all).numpy())


def train_stage(model, loaders, classes, args, device, epochs: int, stage: str,
                unfreeze_last: int, best_state: dict | None, best_score: float):
    """一个训练阶段。每个 epoch 在 model.train() 之后重新锁 BN（关键）。"""
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args.lr_head, args.lr_bb, args.weight_decay)
    scheduler = make_scheduler(optimizer, epochs)

    patience_left = args.patience
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        # 必须在 model.train() 之后调用：否则 model.train() 会把冻结 block 的 BN 也置回 train
        set_frozen_bn_eval(model, unfreeze_last)
        scheduler.step()

        val = evaluate_loader(model, loaders["val"], classes, device)
        history.append({"stage": stage, "epoch": epoch, "train_loss": train_loss,
                        "val_acc": val["acc"], "val_macro_f1": val["macro_f1"],
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"[{stage}] ep{epoch:02d}/{epochs} loss={train_loss:.4f} "
              f"val_acc={val['acc']:.4f} val_f1={val['macro_f1']:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} ({time.time() - t0:.1f}s)")

        if val["macro_f1"] > best_score + 1e-6:          # 按 macro-F1 选模型，不看 acc
            best_score = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
            print(f"  -> 新最优 val macro-F1={best_score:.4f}，已记录权重")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{stage}] 早停：val macro-F1 连续 {args.patience} 轮无提升")
                break
    return best_state, best_score, history


def main() -> None:
    args = parse_args()
    config.ensure_dirs()
    set_seed(args.seed)
    tee = Tee(config.LOG_DIR / f"{args.tag}_train.log")
    sys.stdout = tee
    try:
        device = get_device()
        print(f"== {args.tag} == {now_str()}")
        print(f"device={device} params={vars(args)}")

        loaders, classes, sizes = build_all_loaders(
            args.split, args.batch_size, args.num_workers,
            weighted=args.weighted_sampler, seed=args.seed)
        split_payload = load_json(Path(args.split))
        data_tag = (f"{len(classes)}cls/{sum(sizes.values())}img"
                    f"/seed{split_payload.get('seed', args.seed)}")
        print(f"classes({len(classes)})={classes}")
        print(f"sizes={sizes} data={data_tag}")

        model = build_model(args.arch, len(classes), args.head, args.hidden, args.dropout,
                            pretrained=not args.no_pretrained).to(device)
        freeze_backbone(model)
        print(f"num_params={count_params(model)}")

        # ---- 阶段一：冻结 backbone，只训头 ----
        best_state, best_score, history = train_stage(
            model, loaders, classes, args, device, args.epochs,
            stage="stage1", unfreeze_last=0, best_state=None, best_score=-1.0)

        # ---- 阶段二（可选）：解冻 features[-k:]，重建优化器与 T_max ----
        if args.unfreeze_last > 0:
            model.load_state_dict(best_state)
            cut = unfreeze_last_blocks(model, args.unfreeze_last)
            print(f"阶段二：解冻 features[{cut}:]（{len(list(model.features)) - cut} 个 block）")
            best_state, best_score, hist2 = train_stage(
                model, loaders, classes, args, device, args.unfreeze_epochs,
                stage="stage2", unfreeze_last=args.unfreeze_last,
                best_state=best_state, best_score=best_score)
            history += hist2

        # ---- 用 val 选出的权重做一次最终评估（test 只看这一次）----
        model.load_state_dict(best_state)
        set_frozen_bn_eval(model, args.unfreeze_last)     # 让 BN 统计量与训练时一致
        val = evaluate_loader(model, loaders["val"], classes, device)
        test = evaluate_loader(model, loaders["test"], classes, device)
        print("== 最终评估 ==")
        print(f"val : acc={val['acc']:.4f} macro_f1={val['macro_f1']:.4f}")
        print(f"test: acc={test['acc']:.4f} 95%CI={test['acc_ci95']} "
              f"macro_f1={test['macro_f1']:.4f} top3={test['top3']:.4f}")

        ckpt = {
            "arch": args.arch,
            "classes": classes,
            "img_size": config.IMG_SIZE,
            "mean": list(config.MEAN),
            "std": list(config.STD),
            "head": head_config(args.head, args.arch, args.hidden, args.dropout),
            "temperature": 1.0,        # 由 src/calibrate.py 覆盖后回写
            "threshold": None,         # 同上
            "epochs_trained": len(history),
            "unfreeze_last": args.unfreeze_last,
            "val": {k: val[k] for k in ("n", "acc", "acc_ci95", "macro_f1", "top3", "ece")},
            "test": {k: test[k] for k in ("n", "acc", "acc_ci95", "macro_f1", "top3", "ece")},
            "args": vars(args),
            "env": env_info(),
            "state_dict": best_state,
        }
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{args.tag}_best.pth"
        torch.save(ckpt, ckpt_path)
        print(f"checkpoint: {ckpt_path}")

        save_json(ckpt_dir / f"{args.tag}_history.json",
                  {"history": history, "val": ckpt["val"], "test": ckpt["test"]})
        save_confusion_png(np.asarray(test["confusion"]), classes,
                           Path(args.curves_dir) / f"cm_{args.tag}.png",
                           title=f"{args.tag} / test")

        append_run_csv(Path(args.runs_csv), {
            "run_id": args.tag, "date": now_str(), "git_commit": git_commit(),
            "split": args.split, "data": data_tag,
            "model": args.arch, "head": args.head,
            "freeze": "frozen" if args.unfreeze_last == 0 else f"unfreeze{args.unfreeze_last}",
            "aug": AUG_TAG,
            "optim": f"AdamW lr_h={args.lr_head} lr_b={args.lr_bb} wd={args.weight_decay} "
                     f"ls={args.label_smoothing} bs={args.batch_size} ep={args.epochs}",
            "seed": args.seed,
            "val_acc": round(val["acc"], 4), "val_macro_f1": round(val["macro_f1"], 4),
            "test_acc": round(test["acc"], 4),
            "test_ci_low": test["acc_ci95"][0], "test_ci_high": test["acc_ci95"][1],
            "test_macro_f1": round(test["macro_f1"], 4), "test_top3": round(test["top3"], 4),
            "note": args.note,
        })
        print(f"实验已记录: {args.runs_csv}")
    finally:
        tee.close()


if __name__ == "__main__":
    main()
