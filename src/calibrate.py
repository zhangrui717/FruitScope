"""温度标定 + 拒识阈值标定（计划书 v2 §4.4）。

小数据微调的模型 softmax 过自信（ECE 常常 0.1+），直接把 0.6 当阈值基本永不触发。
本脚本：
1) 在 val 上拟合单一温度 T（最小化 NLL），把 ECE 压下来；
2) 在 val（ID）与 data/ood（OOD）上扫阈值，在"误拒率 <= target"约束下最大化拒识率；
3) 把 temperature / threshold 回写 checkpoint（--update-ckpt）。

    python -m src.calibrate --ckpt ckpt/E03_best.pth --update-ckpt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import build_all_loaders
from src.eval import collect_logits, load_model_from_checkpoint, summarize
from src.ood import build_ood_loader, max_softmax, ood_argmax_report, sweep_threshold
from src.utils import (expected_calibration_error, save_json, set_seed, softmax_np,
                       update_run_csv)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, max_iter: int = 100) -> float:
    """LBFGS 拟合温度；只优化一个标量，几十步就收敛。"""
    z = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)          # 用 log T 保证 T > 0
    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = nll(z / torch.exp(log_t).clamp(min=1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach().clamp(1e-3, 1e3))


def main() -> None:
    ap = argparse.ArgumentParser(description="温度 + 阈值标定")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default=str(config.SPLIT_FILE))
    ap.add_argument("--ood-dir", default=str(config.OOD_DIR))
    ap.add_argument("--target-frr", type=float, default=0.05,
                    help="允许的正常样本误拒率上限（与产品要求对齐）")
    ap.add_argument("--update-ckpt", action="store_true", help="把 T / 阈值写回 checkpoint")
    ap.add_argument("--runs-csv", default=str(config.RUNS_CSV), help="回填 OOD/ECE 到实验表")
    ap.add_argument("--no-runs-update", action="store_true", help="不碰 experiments/runs.csv")
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model_from_checkpoint(args.ckpt, device)
    loaders, classes, sizes = build_all_loaders(args.split, 32, 0)

    val_logits, val_targets = collect_logits(model, loaders["val"], device)
    val_probs_raw = softmax_np(val_logits)
    ece_before = expected_calibration_error(val_probs_raw, val_targets)
    temperature = fit_temperature(val_logits, val_targets)
    ece_after = expected_calibration_error(softmax_np(val_logits / temperature), val_targets)
    print(f"温度 T={temperature:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f} "
          f"(val n={sizes['val']})")
    if temperature >= 999.0 or temperature <= 1.01e-3:
        print(f"[warn] 温度命中上下界（T={temperature:.4g}），说明 logits 几乎没有区分度："
              f"模型没训好 / 类别严重失衡，别拿这次标定结果上线")

    # 阈值：ID 侧用 val 的 MSP，OOD 侧用 data/ood（没有该目录就跳过，阈值留空）
    threshold_payload = {}
    openset_payload: dict = {}
    ood_dir = Path(args.ood_dir)
    if ood_dir.exists() and any(ood_dir.rglob("*.*")):
        ood_logits, _ = collect_logits(model, build_ood_loader(ood_dir, 32, 0), device)
        best = sweep_threshold(max_softmax(val_logits, temperature),
                               max_softmax(ood_logits, temperature), args.target_frr)
        threshold_payload = {"score": "msp", **best}
        print(f"阈值={best['threshold']:.4f} OOD拒识={best['ood_reject']:.4f} "
              f"ID误拒={best['false_reject']:.4f}")
        # 有 other 类时，truth 是"陌生图落到 other"，不是"置信度低于阈值"
        openset_payload = ood_argmax_report(ood_logits, classes)
        if openset_payload["has_other_class"]:
            n_ood = openset_payload["n_ood"]
            print(f"OOD 判为 other: {openset_payload['ood_as_other']:.1%} "
                  f"({int(round(openset_payload['ood_as_other'] * n_ood))}/{n_ood})；"
                  f"另有 {openset_payload['ood_as_fruit']:.1%} 被硬判成水果")
        else:
            print("  (10 类模型没有 other，陌生输入只能靠阈值拒识，见上面的 OOD拒识率)")
        print("  OOD 预测分布: " +
              ", ".join(f"{d['label']}={d['count']}" for d in openset_payload["top_pred"]))
    else:
        print(f"[warn] 未找到 OOD 目录 {ood_dir}，跳过阈值标定（阈值将为 None）")

    test_logits, test_targets = collect_logits(model, loaders["test"], device)
    test_metrics = summarize(test_logits, test_targets, classes, temperature)
    print(f"test(calibrated): acc={test_metrics['acc']:.4f} "
          f"95%CI={test_metrics['acc_ci95']} ECE={test_metrics['ece']:.4f}")

    payload = {"ckpt": args.ckpt, "temperature": temperature,
               "ece_before": ece_before, "ece_after": ece_after,
               "threshold": threshold_payload,
               "open_set": openset_payload,
               "test": {k: test_metrics[k] for k in ("n", "acc", "acc_ci95", "macro_f1", "top3", "ece")}}
    out_json = Path(args.save_json) if args.save_json else \
        config.CKPT_DIR / f"{Path(args.ckpt).stem.replace('_best', '')}_calib.json"
    save_json(out_json, payload)
    print(f"标定结果: {out_json}")

    if not args.no_runs_update:
        tag = Path(args.ckpt).stem.replace("_best", "")
        updates = {"ece": round(ece_after, 4)}
        if threshold_payload:
            updates["ood_reject"] = round(float(threshold_payload["ood_reject"]), 4)
            updates["ood_false_reject"] = round(float(threshold_payload["false_reject"]), 4)
        if update_run_csv(Path(args.runs_csv), tag, updates):
            print(f"已回填 runs.csv: {tag} -> {updates}")
        else:
            print(f"[note] runs.csv 里没有 run_id={tag}，跳过回填")

    if args.update_ckpt:
        ckpt["temperature"] = temperature
        ckpt["threshold"] = threshold_payload.get("threshold", None)
        ckpt["threshold_meta"] = threshold_payload
        ckpt["openset"] = openset_payload
        torch.save(ckpt, args.ckpt)
        print(f"已回写 checkpoint: {args.ckpt} (T={temperature:.3f}, "
              f"threshold={ckpt['threshold']})")


if __name__ == "__main__":
    main()