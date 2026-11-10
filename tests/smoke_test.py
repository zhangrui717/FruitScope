"""端到端冒烟测试：合成小数据集，跑通 准备 -> 训练 -> 评估 -> 校准 -> 推理自检。

真实精度无意义（backbone 随机初始化、只有几十张图），只验证代码链路：
数据划分、BN 锁定、checkpoint 字段、runs.csv 记录、温度/阈值标定、推理预处理。

    python tests/smoke_test.py            # 约 1-3 分钟（CPU 也能跑）
    python tests/smoke_test.py --keep     # 保留临时目录便于排查
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"Apple": (200, 40, 40), "Banana": (230, 210, 60), "Orange": (240, 140, 30)}
OOD_COLOR = (40, 90, 200)          # 蓝色"异物"，只为跑通拒识分支


def make_image(path: Path, color, rng: np.random.Generator, size: int = 96) -> None:
    """纯色底 + 噪声 + 一个随机位置/大小的椭圆主体，保证 pHash 不会把同类图当重复。"""
    scale = rng.uniform(0.8, 1.2)                     # 每张亮度不同 -> pHash 有区分度
    base_color = np.clip(np.array(color, dtype=np.float32) * scale, 0, 255)
    arr = np.zeros((size, size, 3), dtype=np.float32) + base_color
    arr += rng.normal(0, 18, (size, size, 3))
    cx, cy = rng.uniform(0.35, 0.65) * size, rng.uniform(0.35, 0.65) * size
    rx, ry = rng.uniform(0.20, 0.32) * size, rng.uniform(0.16, 0.28) * size
    yy, xx = np.mgrid[0:size, 0:size]
    mask = ((xx - cx) ** 2) / rx ** 2 + ((yy - cy) ** 2) / ry ** 2 < 1
    arr[mask] = base_color * 0.75
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path, quality=92)


def build_dataset(root: Path, per_class: int = 24, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for cls, color in CLASSES.items():
        out = root / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            make_image(out / f"{cls.lower()}_{i:03d}.jpg", color, rng)


def build_ood(root: Path, n: int = 8, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make_image(root / f"odd_{i:02d}.jpg", OOD_COLOR, rng)


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"命令失败 (exit={proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    base = Path(tempfile.mkdtemp(prefix="fruitscope_smoke_"))
    print(f"临时目录: {base}")
    py = sys.executable
    try:
        raw, processed, ood = base / "raw", base / "processed", base / "ood"
        splits, ckpt_dir, runs_csv = base / "splits.json", base / "ckpt", base / "runs.csv"
        build_dataset(raw, args.per_class)
        build_ood(ood)

        # 1) 数据准备（含 dry-run，验证幂等/统计分支）
        report = base / "cleaning_report.md"
        run([py, "scripts/01_data_prepare.py", "--src", str(raw), "--out", str(processed),
             "--split", str(splits), "--report", str(report)])
        run([py, "scripts/01_data_prepare.py", "--src", str(raw), "--out", str(processed),
             "--split", str(splits), "--report", str(report), "--dry-run"])
        if not splits.exists():
            raise SystemExit("splits.json 未生成")

        # 2) 训练（2 epoch，随机初始化，只为跑通流程）
        tag = "S00_smoke"
        run([py, "-m", "src.train", "--tag", tag, "--split", str(splits),
             "--arch", "mobilenet_v3_small", "--epochs", str(args.epochs), "--patience", "2",
             "--batch-size", "8", "--num-workers", "0", "--no-pretrained",
             "--ckpt-dir", str(ckpt_dir), "--runs-csv", str(runs_csv),
             "--curves-dir", str(base / "curves")])
        ckpt = ckpt_dir / f"{tag}_best.pth"
        if not ckpt.exists():
            raise SystemExit("checkpoint 未生成")

        # 3) 评估（混淆矩阵 + JSON）
        run([py, "-m", "src.eval", "--ckpt", str(ckpt), "--split", str(splits),
             "--which", "test", "--num-workers", "0", "--batch-size", "8",
             "--confusion-png", str(base / "cm.png"),
             "--save-json", str(base / "test.json")])

        # 4) 温度 + 阈值标定并回写 checkpoint
        run([py, "-m", "src.calibrate", "--ckpt", str(ckpt), "--split", str(splits),
             "--ood-dir", str(ood), "--update-ckpt", "--save-json", str(base / "calib.json")])

        # 5) 推理自检（--strict 才因错判返回非 0）
        run([py, "-m", "src.predict", "--ckpt", str(ckpt), "--split", str(splits), "--selfcheck"])

        print("\n冒烟测试通过 ✅  （真实训练请用 data/splits.json + 预训练权重）")
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
        else:
            print(f"临时目录已保留: {base}")


if __name__ == "__main__":
    main()
