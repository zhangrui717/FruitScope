"""通用小工具：随机性、指标、实验记录、JSON 读写。

计划书 v2 对应条目：§4.2（set_seed）、§4.4（95% CI / macro-F1 / Top-k / ECE）、
§4.7（runs.csv 自动追加）。
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


# ---------------- 随机性与设备 ----------------

def set_seed(seed: int = 42) -> None:
    """固定全部随机源；配合 cudnn.deterministic 保证同配置可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:      # 允许在没装 torch 的环境里跑纯数据脚本
        pass


def get_device():
    """cuda > mps > cpu。"""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def worker_init_fn(worker_id: int) -> None:
    """DataLoader 子进程也要固定种子，否则增强序列每次都不同。"""
    import torch
    seed = (int(torch.initial_seed()) + worker_id) % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


# ---------------- 指标 ----------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（小测试集必须报，别只写一个百分数）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def topk_correct(logits: np.ndarray, targets: np.ndarray, k: int = 3) -> int:
    """Top-k 命中数（logits: [N, C]）。"""
    if logits.size == 0:
        return 0
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return int((topk == targets[:, None]).any(axis=1).sum())


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """行 = 真实类别，列 = 预测类别。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report(y_true: Sequence[int], y_pred: Sequence[int],
                     class_names: Sequence[str]) -> List[Dict[str, object]]:
    """逐类 precision / recall / f1 / support（不依赖 sklearn）。"""
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    rows: List[Dict[str, object]] = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": name, "precision": prec, "recall": rec,
                     "f1": f1, "support": int(cm[i, :].sum())})
    return rows


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """宏平均 F1 —— 早停与模型选择都用它（acc 在 159 张 val 上分辨率太粗）。"""
    rows = per_class_report(y_true, y_pred, [str(i) for i in range(num_classes)])
    return float(np.mean([float(r["f1"]) for r in rows]))


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """ECE：|acc - conf| 的分箱加权和。"""
    if probs.size == 0:
        return float("nan")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax（推理端 logits/T 之后再 softmax）。"""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------- 记录与 IO ----------------

RUNS_FIELDS = [
    "run_id", "date", "git_commit", "split", "data", "model", "head", "freeze",
    "aug", "optim", "seed", "val_acc", "val_macro_f1",
    "test_acc", "test_ci_low", "test_ci_high", "test_macro_f1", "test_top3",
    "ood_reject", "ood_false_reject", "ece", "note",
]


def append_run_csv(csv_path: Path, row: Dict[str, object]) -> None:
    """把一次实验追加进 experiments/runs.csv（缺失字段自动补空，永不手抄）。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in RUNS_FIELDS})


def update_run_csv(csv_path: Path, run_id: str, updates: Dict[str, object]) -> bool:
    """把后段指标（OOD 拒识率 / ECE）回填到 runs.csv 的对应行；命中返回 True。

    训练时还拿不到标定结果，所以那几列先空着，由 src/calibrate.py 回填，
    避免两处手抄数字对不上。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hit = False
    for row in rows:
        if row.get("run_id") != run_id:
            continue
        for key, value in updates.items():
            if key in RUNS_FIELDS:
                row[key] = "" if value is None else value
        hit = True
    if not hit:
        return False
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def save_json(path: Path, payload: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    """取当前 commit（没装 git / 不是仓库就返回 unknown，不影响训练）。"""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class Tee:
    """同时写 stdout 与日志文件（服务器上即使 nohup 也留档）。"""

    def __init__(self, log_path: Path):
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(log_path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        finally:
            sys.stdout = self.stdout


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_metrics(metrics: Dict[str, object]) -> str:
    """一行式打印，日志好扫。"""
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items())


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def rel_to_root(path: Path) -> str:
    """项目内路径存相对根（换机器/换系统都能用），项目外存绝对 posix 路径。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return path.as_posix()


def iter_files(root: Path, exts: Iterable[str] = IMG_EXTS):
    """递归列出图片（大小写不敏感，跳过 macOS 的 ._ 隐藏文件）。"""
    root = Path(root)
    exts = tuple(e.lower() for e in exts)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("._"):
            yield p
