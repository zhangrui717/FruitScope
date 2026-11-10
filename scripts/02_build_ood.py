"""构建 data/ood 开集测试集：从"蔬果难分"的类别抽图（计划书 v2 §3.5 / §4.4）。

这些类别被 `scripts/01_data_prepare.py --other-exclude` 排除在 other 之外，
对模型来说是真正没见过的类，正好用来：
  1) src/calibrate.py 标定拒识阈值（ID=val, OOD=data/ood）；
  2) 验证上线文案"这不是我认识的水果"（src/predict.py）。

    python scripts/02_build_ood.py --src data/raw/fruit-veg
    python scripts/02_build_ood.py --src data/raw/fruit-veg --per-class 25
    python scripts/02_build_ood.py --src data/raw/fruit-veg --classes tomato lemon --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils import IMG_EXTS, now_str, save_json


def collect(root: Path, names: set) -> Dict[str, List[Path]]:
    """递归按目录名分组（数据集是 train/tomato、test/tomato 这种结构）。

    只看目录直属文件：父目录 train/ 下没有图片，不会被误当成一个 OOD 类别。
    """
    groups: Dict[str, List[Path]] = {}
    for dirpath, _, filenames in os.walk(root):
        name = Path(dirpath).name.strip().lower()
        if name not in names:
            continue
        files = [Path(dirpath) / f for f in sorted(filenames)
                 if Path(f).suffix.lower() in IMG_EXTS and not f.startswith("._")]
        if files:
            groups.setdefault(name, []).extend(files)
    return groups


def load_used_origins(manifest: Path) -> set:
    """读出已被 other 负类占用的原图路径（manifest 的 origin 列）。

    若 other 里已经放了番茄/黄瓜等"蔬果难分"的图片，OOD 必须只用**没被占用**的
    同类别图片，否则就是在测训练集，拒识率会虚高。
    """
    if not manifest.exists():
        return set()
    used = set()
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("class") == config.OTHER_CLASS or row.get("source") == "other":
                used.add(Path(row["origin"]).resolve().as_posix())
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 data/ood 开集测试集")
    ap.add_argument("--src", required=True, help="数据集根目录（含 tomato/ lemon/ 等目录）")
    ap.add_argument("--out", default=str(config.OOD_DIR))
    ap.add_argument("--classes", nargs="*", default=config.OTHER_EXCLUDE_DIRS,
                    help="抽哪些类别当 OOD（默认=other 的排除清单）")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--exclude-from", default=str(config.DATA_DIR / "manifest_other.csv"),
                    help="从该 manifest 里读出已被 other 占用的原图并跳过（保证 OOD 与训练不重叠）")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--clean", action="store_true",
                    help="先清空输出目录（否则旧的 OOD 图会混进新集合，污染标定）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"源目录不存在: {src}")

    wanted = {c.strip().lower() for c in args.classes}
    groups = collect(src, wanted)
    if not groups:
        raise SystemExit(f"在 {src} 下没找到这些目录: {sorted(wanted)}")
    missing = sorted(wanted - set(groups))
    if missing:
        print(f"[warn] 数据集中不存在这些目录（跳过）: {missing}")

    out = Path(args.out)
    used = load_used_origins(Path(args.exclude_from))
    print(f"已排除 other 训练用图: {len(used)} 张（来自 {args.exclude_from}）")
    picked: Dict[str, List[Path]] = {}
    for name in sorted(groups):
        files = [f for f in sorted(groups[name]) if f.resolve().as_posix() not in used]
        skipped = len(groups[name]) - len(files)
        if not files:
            print(f"  {name:<14} 全部被 other 占用，跳过（{skipped} 张）")
            continue
        take = min(args.per_class, len(files))
        idx = sorted(rng.choice(len(files), size=take, replace=False))
        picked[name] = [files[i] for i in idx]
        extra = f"，跳过已占用 {skipped}" if skipped else ""
        print(f"  {name:<14} 可用 {len(files):4d} -> 取 {take:3d}{extra}")

    total = sum(len(v) for v in picked.values())
    print(f"OOD 合计 {total} 张，覆盖 {len(picked)} 类；来源 {src}")
    if args.dry_run:
        print("[dry-run] 不落盘")
        return

    if out.exists() and any(out.iterdir()):
        if args.clean:
            shutil.rmtree(out)
            print(f"已清空旧 OOD 目录: {out}")
        else:
            print(f"[warn] {out} 非空：旧图会混进本次 OOD 集合，建议加 --clean")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, files in picked.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(files, start=1):
            dst = out / name / f"{name.replace(' ', '_')}_{i:04d}{p.suffix.lower()}"
            shutil.copy2(p, dst)
            rows.append({"file": dst.as_posix(), "class": name, "origin": str(p)})

    manifest = out / "_manifest.json"
    save_json(manifest, {"created": now_str(), "source": str(src),
                         "per_class": args.per_class, "classes": sorted(picked),
                         "count": len(rows), "items": rows})
    print(f"OOD 目录: {out}（{len(rows)} 张）")
    print(f"清单: {manifest}")
    print("下一步: python -m src.calibrate --ckpt ckpt/<tag>_best.pth --update-ckpt")


if __name__ == "__main__":
    main()