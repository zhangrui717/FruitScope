"""数据准备：合并主数据集 -> pHash 去重 -> 70/15/15 重划分 -> splits.json + 清洗报告。

计划书 v2 §3.4 / §3.5 / §8：
- 官方 test 只有约 10 张/类（85% 的 95% CI 约 ±7%），必须合并后重划分；
- 同源图片常跨 train/val/test 重复，先去重、划分后再查一次泄漏；
- 产物：data/processed/<class>/<img>、data/splits.json（进 Git）、data/manifest.csv、
        docs/cleaning_report.md。

用法：
    # 只有主数据集（10 类，约 1000 张）
    python scripts/01_data_prepare.py --src data/raw/fruit-recognition

    # 叠加 Fruits-360 子集（每类最多 300 张，文件夹名必须写在 data/fruits360_map.json 里）
    python scripts/01_data_prepare.py --src data/raw/fruit-recognition ^
        --fruits360 data/raw/fruits-360 --f360-per-class 300

    # 再抽负样本成 other 类（11 类训练）
    python scripts/01_data_prepare.py --src ... --other-from data/raw/vegetables --other-per-class 60

    # 只统计不落盘
    python scripts/01_data_prepare.py --src ... --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dedup import cross_split_leaks, dedupe_by_phash, phash
from src.utils import IMG_EXTS, iter_files, now_str, save_json

# 目录名别名：主数据集里是 "Grape"（单数），Fruits-360 里是 "Grape Blue 1"
ALIASES = {
    "apple": "apple", "apples": "apple",
    "banana": "banana", "bananas": "banana",
    "grape": "grapes", "grapes": "grapes",
    "kiwi": "kiwi", "kiwifruit": "kiwi",
    "mango": "mango", "mangos": "mango", "mangoes": "mango",
    "orange": "orange", "oranges": "orange",
    "pear": "pear", "pears": "pear",
    "pineapple": "pineapple", "pineapples": "pineapple",
    "pomegranate": "pomegranate", "pomegranates": "pomegranate",
    "watermelon": "watermelon", "watermelons": "watermelon",
}


# other 负类默认排除清单（计划书 §4.4 复核项）：
# 这些"蔬果难分"的类别如果被教成 other，模型的类别边界会变糊；
# 它们正好留作 data/ood 的开集测试集（见 scripts/02_build_ood.py）。
DEFAULT_OTHER_EXCLUDE = config.OTHER_EXCLUDE_DIRS


def normalize_name(name: str) -> str | None:
    return ALIASES.get(name.strip().lower())


def scan_class_dirs(root: Path) -> Dict[str, List[Path]]:
    """在任意层级找到"名字等于水果类"的目录，收集其下全部图片。

    兼容 data/raw/Fruit/{train,validation,test}/Apple 与 data/raw/Fruit/Apple 两种结构。
    """
    found: Dict[str, List[Path]] = {}
    for dirpath, dirnames, _ in os.walk(root):
        cls = normalize_name(Path(dirpath).name)
        if cls is None:
            continue
        found.setdefault(cls, []).extend(iter_files(Path(dirpath)))
        dirnames[:] = []            # 该类目录内部不再下钻，避免重复计数
    return found


def collect_fruits360(root: Path, map_file: Path, per_class: int, rng) -> Tuple[Dict[str, List[Path]], dict]:
    """按映射表收集 Fruits-360 子集；只认显式目录名，绝不做子串匹配。

    （Fruits-360 里有 `Apple ...` 与 `Pineapple ...`，用 "apple" 做子串匹配会把菠萝算成苹果。）
    """
    if not root.exists():
        print(f"[warn] Fruits-360 目录不存在: {root}，跳过")
        return {}, {}
    if not map_file.exists():
        print(f"[warn] 缺少映射表 {map_file}，跳过 Fruits-360")
        return {}, {}

    payload = json.loads(map_file.read_text(encoding="utf-8"))
    mapping = payload.get("mapping", payload)
    dirs = {p.name.strip().lower(): p for p in root.iterdir() if p.is_dir()}

    out: Dict[str, List[Path]] = {}
    used, unmatched = set(), []
    for cls, names in mapping.items():
        if cls.startswith("_") or not isinstance(names, list):
            continue
        pool: List[Path] = []
        for name in names:
            d = dirs.get(name.strip().lower())
            if d is None:
                unmatched.append(name)
                continue
            pool.extend(iter_files(d))
            used.add(d.name.lower())
        if not pool:
            continue
        take = min(per_class, len(pool))
        picked = rng.choice(len(pool), size=take, replace=False)
        out[cls] = [pool[i] for i in sorted(picked)]

    unused = sorted(set(dirs) - used)
    report = {"per_class": per_class, "picked": {k: len(v) for k, v in out.items()},
              "unmatched_names": unmatched[:50], "unused_dirs": unused[:50],
              "unused_dir_count": len(unused)}
    if unmatched:
        print(f"[warn] 映射表里这些目录不存在（Fruits-360 版本可能不同）: {unmatched[:10]}")
    if unused:
        print(f"[note] Fruits-360 有 {len(unused)} 个目录未被使用，示例: {unused[:8]}")
    return out, report


def split_class(paths: List[Path], rng, val_ratio: float, test_ratio: float
                ) -> Dict[str, List[Path]]:
    """单类内部分层切分：保证 val/test 至少各 1 张（样本够时），train 不为空。"""
    n = len(paths)
    order = list(range(n))
    rng.shuffle(order)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 10:
        n_test, n_val = max(1, n_test), max(1, n_val)
    n_train = n - n_val - n_test
    if n_train < 1:                      # 极少样本时退化，但要保住 train
        n_train = max(1, n - 2)
        n_val = 1 if n - n_train >= 1 else 0
        n_test = n - n_train - n_val
    idx = {"train": order[:n_train],
           "val": order[n_train:n_train + n_val],
           "test": order[n_train + n_val:]}
    return {name: [paths[i] for i in ids] for name, ids in idx.items()}


def collect_named_dirs(root: Path) -> Dict[str, List[Path]]:
    """递归找出所有"直接含图片"的目录，按目录名分组。

    数据集是 train/tomato、validation/tomato、test/tomato 这种多层结构，
    直接 iterdir() 只会看到 train/validation/test 三个"类"，所以按目录名合并；
    但必须只看**直属文件**（iter_files 是递归的，会把 train/ 当成一个巨大的负类）。
    """
    groups: Dict[str, List[Path]] = {}
    for dirpath, _, filenames in os.walk(root):
        files = [Path(dirpath) / f for f in sorted(filenames)
                 if Path(f).suffix.lower() in IMG_EXTS and not f.startswith("._")]
        if not files:
            continue
        groups.setdefault(Path(dirpath).name.strip().lower(), []).extend(files)
    return groups


def other_pool(src_dirs: List[Path], per_class: int, rng,
               exclude: Iterable[str] | None = None,
               total: int = 0) -> Tuple[List[Path], dict]:
    """负类池：从非目标目录抽样，满足计划书 §4.4 的四条约束。

    1) 目标水果目录一律跳过（按 normalize_name 判断，不是子串匹配）；
    2) 排除"蔬果难分"的类别（番茄/黄瓜/甜椒/辣椒/茄子/牛油果/柠檬），
       它们留给 data/ood 做开集测试，否则 other 的边界会被教糊；
    3) `total > 0` 时按目录数均分额度，保证 other 总量落在 300–500 量级；
    4) 返回每个目录实际取了几张，写进清洗报告，便于复核有没有单类独大。
    """
    exclude = {e.strip().lower() for e in (exclude or [])}
    groups: Dict[str, List[Path]] = {}
    for src in src_dirs:
        if not src.exists():
            print(f"[warn] 负类目录不存在: {src}")
            continue
        for name, files in collect_named_dirs(src).items():
            if normalize_name(name):
                continue                     # 目标水果不进负类
            if name in exclude:
                continue                     # 蔬果难分，留给 data/ood
            groups.setdefault(name, []).extend(files)

    if not groups:
        print("[warn] other 没找到任何可用目录（检查 --other-from / --other-exclude）")
        return [], {"dirs": [], "used": {}, "per_class_cap": 0, "total": 0,
                    "excluded": sorted(exclude), "target_total": total}

    names = sorted(groups)
    per = per_class
    if total and total > 0:                  # 总量优先：按目录数均分
        per = max(1, min(per_class, total // len(names)))

    pool: List[Path] = []
    used: Dict[str, int] = {}
    for name in names:
        files = sorted(groups[name])
        take = min(per, len(files))
        picked = rng.choice(len(files), size=take, replace=False)
        pool.extend(files[i] for i in sorted(picked))
        used[name] = take

    if total and len(pool) > total:          # 二次收紧（各类样本数不均时）
        idx = sorted(rng.choice(len(pool), size=total, replace=False))
        pool = [pool[i] for i in idx]

    info = {"dirs": names, "used": used, "per_class_cap": per,
            "total": len(pool), "excluded": sorted(exclude), "target_total": total}
    return pool, info


def rel_to_root(path: Path) -> str:
    """splits.json 里存相对项目根的路径，换机器/换系统都能用。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(config.ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def short_path(path: str) -> str:
    """报告里显示 raw/train/apple/Image_1.jpg 这样的末三级路径。

    只显示文件名看不出"跨 train/test 的重复图"，这是核查去重是否正确的最小信息。
    """
    return "/".join(Path(path).parts[-3:])


def write_report(path: Path, ctx: dict) -> None:
    """生成 docs/cleaning_report.md（计划书 §3.5 要求留档清洗过程）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    classes: List[str] = ctx["classes"]
    counts: Dict[str, Dict[str, int]] = ctx["counts"]
    lines = [
        "# 数据清洗与划分报告",
        "",
        f"- 生成时间：{now_str()}",
        f"- 随机种子：{ctx['seed']}（同参数重跑应得到同一划分）",
        f"- 主数据集：`{ctx['src']}`",
        f"- Fruits-360：`{ctx['fruits360']}`（每类最多 {ctx['f360_per_class']} 张）",
        f"- 负类来源：{ctx['other_from'] or '未使用'}",
        f"- pHash 去重阈值：汉明距离 <= {ctx['phash_threshold']}",
        "",
        "## 1. 各类张数（去重后 / 划分后）",
        "",
        "| 类别 | 中文 | 去重后 | train | val | test |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cls in classes:
        lines.append(f"| {cls} | {config.class_cn(cls)} | {ctx['kept_counts'].get(cls, 0)} | "
                     f"{counts['train'][cls]} | {counts['val'][cls]} | {counts['test'][cls]} |")
    total = {name: sum(counts[name].values()) for name in counts}
    lines.append(f"| **合计** | - | **{sum(total.values())}** | **{total['train']}** | "
                 f"**{total['val']}** | **{total['test']}** |")

    other_info = ctx.get("other_info") or {}
    if other_info.get("used"):
        used = other_info["used"]
        n_train = max(1, total["train"])
        share = counts["train"].get(config.OTHER_CLASS, 0) / n_train
        lines += ["", "## 1b. other 负类构成", "",
                  f"- 总量：**{other_info['total']}** 张（目标 {other_info.get('target_total') or '不限'}，"
                  f"每类上限 {other_info['per_class_cap']}），覆盖 {len(used)} 个目录；",
                  f"- 训练集占比：**{share:.1%}**（超过 40% 就调小 `--other-total`，"
                  f"并务必开 `--weighted-sampler`）；",
                  f"- 已排除（蔬果难分，留作 `data/ood` 开集测试）："
                  f"{', '.join(other_info.get('excluded', [])) or '无'}", "",
                  "| 负类目录 | 取用张数 |", "| --- | --- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(used.items())]
        lines.append("")

    lines += ["", "## 2. 去重", "",
              f"- 原始图片 {ctx['raw_total']} 张，pHash 丢弃 {len(ctx['dropped'])} 张，"
              f"保留 {ctx['raw_total'] - len(ctx['dropped'])} 张。",
              f"- 按来源统计：{ctx['dropped_by_source']}", ""]
    if ctx["dropped"]:
        lines += ["| 类别 | 丢弃 | 保留的近似图 | 汉明距离 |", "| --- | --- | --- | --- |"]
        for row in ctx["dropped"][:20]:
            lines.append(f"| {row['class']} | `{short_path(row['dropped'])}` | "
                         f"`{short_path(row['kept'])}` | {row['distance']} |")
        if len(ctx["dropped"]) > 20:
            lines.append(f"| ... | 其余 {len(ctx['dropped']) - 20} 条见控制台输出 | | |")
        lines.append("")

    leaks = ctx["leaks"]
    lines += ["## 3. 跨划分泄漏检查", "",
              (f"- 发现 {len(leaks)} 对近重复（val/test 与 train 汉明距离 <= "
               f"{ctx['phash_threshold']}），需人工确认或调低阈值重跑。" if leaks
               else "- 未发现 val/test 与 train 的近重复图 ✅"), ""]
    if leaks:
        lines += ["| 划分 | 图片 | 与 train 中 | 距离 |", "| --- | --- | --- | --- |"]
        for row in leaks[:15]:
            lines.append(f"| {row['split']} | `{Path(row['image']).name}` | "
                         f"`{Path(row['train_match']).name}` | {row['distance']} |")
        lines.append("")

    lines += ["## 4. 结论与待办", "",
              "- 划分比例 70/15/15，样本够时 val/test 每类至少 1 张；小类指标波动大，看 95% CI。",
              "- 若启用 Fruits-360，检查映射表未收录的目录（脚本会打印），避免漏类/错类。",
              "- 下一步：`python -m src.train --tag E01_mbv3l_linear`。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"清洗报告: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="数据准备：合并 + pHash 去重 + 重划分")
    ap.add_argument("--src", required=True, help="主数据集根目录（含 Apple/ Banana/ ... 目录）")
    ap.add_argument("--out", default=str(config.PROCESSED_DIR))
    ap.add_argument("--split", default=str(config.SPLIT_FILE))
    ap.add_argument("--fruits360", default=None, help="Fruits-360 根目录（可选）")
    ap.add_argument("--f360-map", default=str(config.FRUITS360_MAP_FILE))
    ap.add_argument("--f360-per-class", type=int, default=0, help="每类最多取多少张，0=不用")
    ap.add_argument("--other-from", nargs="*", default=None, help="负类目录（蔬菜/非水果照片）")
    ap.add_argument("--other-per-class", type=int, default=60, help="每个负类目录最多取几张")
    ap.add_argument("--other-total", type=int, default=0,
                    help="负类总张数上限（>0 时按目录数均分额度；建议 300–500）")
    ap.add_argument("--other-exclude", nargs="*", default=DEFAULT_OTHER_EXCLUDE,
                    help="要从负类里排除的目录名（蔬果难分，默认见 DEFAULT_OTHER_EXCLUDE）")
    ap.add_argument("--other-include-ambiguous", action="store_true",
                    help="把蔬果难分类别也纳入 other（OOD 改用同类别中未被占用的图片，"
                         "见 scripts/02_build_ood.py --exclude-from）")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--phash-threshold", type=int, default=8)
    ap.add_argument("--report", default=str(config.DOCS_DIR / "cleaning_report.md"),
                    help="清洗报告路径（冒烟测试指向临时目录）")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"源目录不存在: {src}")
    rng = np.random.default_rng(args.seed)

    print(f"扫描主数据集: {src}")
    main_files = scan_class_dirs(src)
    if not main_files:
        raise SystemExit("没有找到任何水果类目录（检查目录名是否为 Apple/Banana/Grape/...）")
    print("主数据集各类张数: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(main_files.items())))

    f360_files: Dict[str, List[Path]] = {}
    if args.fruits360 and args.f360_per_class > 0:
        f360_files, _ = collect_fruits360(
            Path(args.fruits360), Path(args.f360_map), args.f360_per_class, rng)
        print("Fruits-360 取用: " +
              (", ".join(f"{k}={len(v)}" for k, v in sorted(f360_files.items())) or "无"))

    pools: Dict[str, List[Tuple[str, Path]]] = {}
    source_of: Dict[str, str] = {}
    for cls, paths in main_files.items():
        for p in paths:
            pools.setdefault(cls, []).append((str(p), p))
            source_of[str(p)] = "main"
    for cls, paths in f360_files.items():
        for p in paths:
            pools.setdefault(cls, []).append((str(p), p))
            source_of[str(p)] = "fruits360"
    other_info: dict = {}
    if args.other_from:
        exclude = [] if args.other_include_ambiguous else args.other_exclude
        if args.other_include_ambiguous:
            print("[note] --other-include-ambiguous：蔬果难分类别也进 other；"
                  "OOD 集请用 02_build_ood.py --exclude-from data/manifest.csv 保证不重叠")
        others, other_info = other_pool([Path(p) for p in args.other_from],
                                        args.other_per_class, rng,
                                        exclude=exclude, total=args.other_total)
        for p in others:
            pools.setdefault(config.OTHER_CLASS, []).append((str(p), p))
            source_of[str(p)] = "other"
        print(f"负类 other: {len(others)} 张，来自 {len(other_info.get('used', {}))} 个目录"
              f"（每类<= {other_info.get('per_class_cap')}）")
        print("  other 目录: " + ", ".join(f"{k}={v}" for k, v in sorted(other_info.get("used", {}).items())))
        print("  已排除(留给 OOD): " + ", ".join(other_info.get("excluded", [])))

    print(f"按类去重（pHash, 阈值 {args.phash_threshold}）:")
    kept: Dict[str, List[Path]] = {}
    dropped: List[dict] = []
    raw_total = sum(len(v) for v in pools.values())
    for cls in sorted(pools):
        items = sorted(pools[cls], key=lambda kv: kv[0])
        keep_items, drop = dedupe_by_phash(items, args.phash_threshold)
        kept[cls] = [p for _, p in keep_items]
        for d in drop:
            dropped.append({"class": cls, "source": source_of.get(d["dropped"], "?"), **d})
        print(f"  {cls:<12} {len(items):5d} -> {len(kept[cls]):5d}（丢弃 {len(drop)}）")

    classes = [c for c in sorted(kept) if c != config.OTHER_CLASS]
    if config.OTHER_CLASS in kept:
        classes.append(config.OTHER_CLASS)
    kept = {c: kept[c] for c in classes}
    kept_counts = {c: len(kept[c]) for c in classes}

    splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    label_of: Dict[Path, str] = {}
    for cls in classes:
        part = split_class(kept[cls], rng, args.val_ratio, args.test_ratio)
        for name, plist in part.items():
            splits[name].extend(plist)
            for p in plist:
                label_of[p] = cls
    counts = {name: {cls: sum(1 for p in splits[name] if label_of[p] == cls) for cls in classes}
              for name in splits}

    print("\n划分结果:")
    for name in ("train", "val", "test"):
        print(f"  {name:<5} {len(splits[name]):5d} 张 | " +
              ", ".join(f"{c}={counts[name][c]}" for c in classes))
    # ---- 失衡检查：other 单类体量必须可控（计划书 §4.4 复核项）----
    train_counts = counts["train"]
    if train_counts:
        n_train = max(1, sum(train_counts.values()))
        hi, lo = max(train_counts.values()), max(1, min(train_counts.values()))
        other_share = train_counts.get(config.OTHER_CLASS, 0) / n_train
        print(f"训练集 最大/最小类 = {hi}/{lo} = {hi / lo:.2f}x；other 占比 = {other_share:.1%}")
        if hi / lo > 2.0:
            print(f"[warn] 类别失衡 {hi / lo:.1f}x -> 训练时务必加 --weighted-sampler")
        if other_share > 0.40:
            print(f"[warn] other 占训练集 {other_share:.1%} (>40%) -> 调小 --other-total")
        if config.OTHER_CLASS in train_counts and other_share < 0.10:
            print(f"[note] other 只占 {other_share:.1%}，负类信号偏弱，可适当加大 --other-total")

    if args.dry_run:
        print("[dry-run] 不写任何文件")
        return

    # ---- 落盘：processed/<class>/<class>_00001.jpg ----
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name_of: Dict[Path, Path] = {}
    for cls in classes:
        (out / cls).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(kept[cls]):
            name_of[p] = out / cls / f"{cls}_{i:05d}{p.suffix.lower()}"

    records: Dict[str, List[dict]] = {name: [] for name in splits}
    manifest_rows: List[dict] = []
    hashes: Dict[str, int] = {}
    for name in ("train", "val", "test"):
        for p in sorted(splits[name], key=lambda x: (label_of[x], str(x))):
            dst = name_of[p]
            shutil.copy2(p, dst)
            rel = rel_to_root(dst)
            records[name].append({"path": rel, "label": label_of[p]})
            manifest_rows.append({"file": rel, "class": label_of[p], "split": name,
                                  "source": source_of.get(str(p), "?"), "origin": str(p)})
            hashes[rel] = phash(dst)

    payload = {
        "created": now_str(),
        "seed": args.seed,
        "ratios": [round(1 - args.val_ratio - args.test_ratio, 4),
                   args.val_ratio, args.test_ratio],
        "source": {"main": str(src), "fruits360": args.fruits360, "other": args.other_from},
        "dedupe": {"method": "phash", "threshold": args.phash_threshold, "dropped": len(dropped)},
        "classes": classes,
        "counts": counts,
        "splits": records,
    }
    save_json(Path(args.split), payload)
    print(f"划分文件: {args.split}")

    stem = Path(args.split).stem            # splits / splits_other
    manifest_path = Path(args.split).parent / (
        "manifest.csv" if stem == "splits" else f"manifest_{stem.split('_', 1)[-1]}.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "class", "split", "source", "origin"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"清单: {manifest_path}")

    # ---- 划分后再查一次泄漏 ----
    leaks = cross_split_leaks({k: [r["path"] for r in v] for k, v in records.items()},
                              hashes, args.phash_threshold)
    print(f"跨划分泄漏检查: {'发现 ' + str(len(leaks)) + ' 对近重复' if leaks else '未发现'}")

    dropped_by_source: Dict[str, int] = {}
    for row in dropped:
        dropped_by_source[row["source"]] = dropped_by_source.get(row["source"], 0) + 1

    write_report(Path(args.report), {
        "seed": args.seed,
        "src": src,
        "fruits360": args.fruits360 or "未使用",
        "f360_per_class": args.f360_per_class,
        "other_from": args.other_from,
        "other_info": other_info,
        "phash_threshold": args.phash_threshold,
        "classes": classes,
        "counts": counts,
        "kept_counts": kept_counts,
        "raw_total": raw_total,
        "dropped": dropped,
        "dropped_by_source": dropped_by_source,
        "leaks": leaks,
    })
    print("完成。下一步: python -m src.train --tag E01_mbv3l_linear --epochs 30")


if __name__ == "__main__":
    main()
