"""生成终端演示用的固定素材：`demo/samples/` + `demo/manifest.json`。

演示要"看起来像真的"，就用**测试集 / OOD 集**的图（绝不碰 train），并且由当前
checkpoint 现场判定后挑选，保证：

1. 10 个水果类各至少 1 张"判对"的案例；
2. 至少 1 张 other（非目标，如番茄）；
3. 至少 1 张低置信度触发拒识；
4. 尽量带 1 张已知的边界误判（柠檬→橙子），诚实展示局限。

同一 checkpoint + 同一划分 ⇒ 同一批图（挑选完全确定，无随机）。不要手工往 demo/samples 丢文件。

    python scripts/05_build_demo.py                       # 默认 E04 + splits_other_broad.json
    python scripts/05_build_demo.py --ckpt ckpt/E01_mbv3l_linear_best.pth \
        --split data/splits.json                          # 10 类版（没有 other 案例）
    python scripts/05_build_demo.py --dry-run             # 只看会挑哪些图，不落盘
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.predict import Predictor
from src.utils import IMG_EXTS, load_json, now_str, rel_to_root, save_json

KIND_CN = {
    "correct": "正确命中",
    "other": "非目标(other)",
    "reject": "低置信拒识",
    "misclass": "边界误判",
}


# ---------------- 小工具 ----------------

def abs_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else config.ROOT / p


def default_split() -> Path:
    """优先宽口径 11 类，其次窄口径，最后 10 类。"""
    for cand in (config.SPLIT_OTHER_BROAD_FILE, config.SPLIT_OTHER_FILE, config.SPLIT_FILE):
        if cand.exists():
            return cand
    raise SystemExit(
        "找不到任何划分文件（data/splits*.json）。\n"
        "先准备数据：python scripts/01_data_prepare.py --src data/raw/fruit-veg "
        "--out data/processed --split data/splits.json")


def load_origin_map(split_file: Path) -> Dict[str, str]:
    """processed 路径 -> 原始图片路径（01_data_prepare.py 写的 manifest_*.csv）。"""
    stem = split_file.stem
    name = "manifest.csv" if stem == "splits" else f"manifest_{stem.split('_', 1)[-1]}.csv"
    path = split_file.parent / name
    if not path.exists():
        print(f"[warn] 找不到 {path}，说明里将无法标注蔬菜原名")
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: row.get("origin", "") for row in csv.DictReader(f)}


def load_ood(ood_dir: Path) -> List[dict]:
    """OOD 目录 -> [{path, label, source}]；优先读 02_build_ood.py 写的 _manifest.json。"""
    if not ood_dir.exists():
        return []
    items: List[dict] = []
    meta = ood_dir / "_manifest.json"
    if meta.exists():
        for row in load_json(meta).get("items", []):
            items.append({"path": abs_path(row["file"]), "label": row["class"], "source": "ood"})
    else:
        for p in sorted(ood_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("."):
                items.append({"path": p, "label": p.parent.name, "source": "ood"})
    return [it for it in items if it["path"].exists()]


def judge(predictor: Predictor, items: List[dict]) -> List[dict]:
    """逐张推理，附上 state（fruit / other / reject）——与正式推理同一个 Predictor。"""
    out: List[dict] = []
    for it in items:
        try:
            result = predictor.predict(it["path"])
        except Exception as exc:                     # 单张坏图不该让整批失败
            print(f"[warn] 跳过无法读取的图片 {it['path'].name}: {exc}")
            continue
        pred = result["topk"][0]["label"]
        state = "reject" if result["rejected"] else (
            "other" if pred == config.OTHER_CLASS else "fruit")
        out.append({**it, "pred": pred, "prob": float(result["confidence"]),
                    "state": state, "result": result})
    return out


# ---------------- 挑选策略 ----------------

def pick_correct(cands: List[dict], model_classes: List[str]) -> List[dict]:
    """每个水果类挑一张判对的（置信度最高）；判错就如实降级成 misclass。"""
    picked: List[dict] = []
    for cls in model_classes:
        if cls == config.OTHER_CLASS:
            continue
        pool = [c for c in cands if c["label"] == cls and c["state"] == "fruit" and c["pred"] == cls]
        if not pool:
            near = sorted([c for c in cands if c["label"] == cls],
                          key=lambda c: (-c["prob"], str(c["path"])))
            if not near:
                print(f"[warn] 测试集里没有 {cls} 的图片，跳过该类别")
                continue
            best = near[0]
            print(f"[warn] {cls} 没有判对的测试图（最好一张判成 {best['pred']} "
                  f"{best['prob']:.3f}），按误判案例收录")
            picked.append({"kind": "misclass", "cand": best,
                           "note": f"测试集 {cls} 未被判对（判成 {best['pred']} {best['prob']:.3f}），如实展示"})
            continue
        best = sorted(pool, key=lambda c: (-c["prob"], str(c["path"])))[0]
        picked.append({"kind": "correct", "cand": best,
                       "note": f"测试集 {cls} 图，判对，置信度 {best['prob']:.3f}"})
    return picked


def pick_other(cands: List[dict], origin_map: Dict[str, str], n: int) -> List[dict]:
    pool = [c for c in cands if c["label"] == config.OTHER_CLASS and c["state"] == "other"]
    pool.sort(key=lambda c: (-c["prob"], str(c["path"])))
    if not pool:
        print("[warn] 测试集里没有一张 other 被判成 other（负类信号没学到？）")
        return []
    picked = []
    for c in pool[:n]:
        origin = origin_map.get(rel_to_root(c["path"]), "")
        veg = Path(origin).parent.name if origin else "蔬菜/非目标"
        picked.append({"kind": "other", "cand": c,
                       "note": f"测试集 other（原图来自 {veg}），判成 other，置信度 {c['prob']:.3f}"})
    return picked


def pick_reject(cands: List[dict], ood: List[dict], threshold) -> List[dict]:
    """优先测试集里真的被拒识的（最低置信度），其次 OOD 集。"""
    for pool, tag in ((cands, "测试集"), (ood, "OOD 集")):
        rejected = [c for c in pool if c["state"] == "reject"]
        if rejected:
            c = sorted(rejected, key=lambda x: (x["prob"], str(x["path"])))[0]
            thr = f"{float(threshold):.3f}" if threshold is not None else "未标定"
            return [{"kind": "reject", "cand": c,
                     "note": f"{tag}里置信度最低的一张（{c['prob']:.3f} < 阈值 {thr}），"
                             f"触发'未识别为支持的水果'"}]
    print("[warn] 没有找到触发拒识的图片（阈值可能未标定，或测试集太简单）；"
          "重新跑 python -m src.calibrate --ckpt <ckpt> --update-ckpt 后再生成")
    return []


def pick_misclass(ood: List[dict], cands: List[dict], used: set) -> List[dict]:
    """已知局限：OOD 的柠檬常被判成橙子。优先 OOD，其次测试集误判。"""
    for pool, tag in ((ood, "OOD 集"), (cands, "测试集")):
        bad = [c for c in pool
               if c["state"] == "fruit" and c["pred"] != c["label"]
               and c["path"].resolve() not in used]
        if bad:
            c = sorted(bad, key=lambda x: (-x["prob"], str(x["path"])))[0]
            why = "模型未训练过该类别" if tag == "OOD 集" else "该类别训练过但仍判错"
            return [{"kind": "misclass", "cand": c,
                     "note": f"{tag}原图是 {c['label']}（{why}），被判成 {c['pred']}，"
                             f"置信度 {c['prob']:.3f}"}]
    print("[warn] 没有找到误判样本（模型在当前集上表现完美），演示将只含正确/other/拒识三态")
    return []



# ---------------- 主流程 ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="生成终端演示素材（demo/samples + manifest.json）")
    ap.add_argument("--ckpt", default=str(config.DEMO_DEFAULT_CKPT))
    ap.add_argument("--split", default=None,
                    help="默认取 data/splits_other_broad.json → splits_other → splits")
    ap.add_argument("--ood-dir", default=str(config.OOD_DIR))
    ap.add_argument("--out-dir", default=str(config.DEMO_SAMPLES_DIR))
    ap.add_argument("--manifest", default=str(config.DEMO_MANIFEST))
    ap.add_argument("--other-cases", type=int, default=2, help="other 案例张数")
    ap.add_argument("--keep-stale", action="store_true", help="保留 samples 目录里的旧文件")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        available = sorted(p.name for p in config.CKPT_DIR.glob("*.pth"))
        raise SystemExit(
            f"找不到 checkpoint: {ckpt}\n"
            f"ckpt/ 下现有: {available or '空'}\n"
            "先训练：python -m src.train --tag E04_broad_unfreeze3 "
            "--split data/splits_other_broad.json --epochs 30 --unfreeze-last 3 "
            "--unfreeze-epochs 15 --weighted-sampler")

    split_file = Path(args.split) if args.split else default_split()
    if not split_file.exists():
        raise SystemExit(f"找不到划分文件: {split_file}")

    test_records = (load_json(split_file).get("splits") or {}).get("test") or []
    if not test_records:
        raise SystemExit(f"{split_file} 里没有 test 划分，无法生成演示素材")

    predictor = Predictor(ckpt, topk=3)
    origin_map = load_origin_map(split_file)

    test_items = [{"path": abs_path(r["path"]), "label": r["label"], "source": "test"}
                  for r in test_records if abs_path(r["path"]).exists()]
    missing = len(test_records) - len(test_items)
    if missing:
        print(f"[warn] {missing} 张测试图不存在（processed 目录被清过？），已跳过")
    if not test_items:
        raise SystemExit(f"{split_file} 的测试图一张都找不到，请先重建 data/processed*")

    thr = "None" if predictor.threshold is None else f"{float(predictor.threshold):.3f}"
    print(f"checkpoint: {ckpt.name}（{len(predictor.classes)} 类，T={predictor.temperature:.3f}，"
          f"阈值={thr}）")
    print(f"划分文件: {rel_to_root(split_file)}，test {len(test_items)} 张")

    cands = judge(predictor, test_items)
    ood = judge(predictor, load_ood(Path(args.ood_dir)))
    print(f"OOD 图: {len(ood)} 张（{rel_to_root(Path(args.ood_dir))}）")

    selected = pick_correct(cands, predictor.classes)
    if config.OTHER_CLASS in predictor.classes:
        selected += pick_other(cands, origin_map, args.other_cases)
    else:
        print("[note] 该 checkpoint 没有 other 类（10 类模型），跳过 other 案例；"
              "要演三态请用 E04/E03 等 11 类 checkpoint")
    selected += pick_reject(cands, ood, predictor.threshold)
    used = {s["cand"]["path"].resolve() for s in selected}
    selected += pick_misclass(ood, cands, used)

    # 去重：同一张原图不能同时当"正确"和"误判"案例
    unique, seen = [], set()
    for item in selected:
        key = item["cand"]["path"].resolve()
        if key in seen:
            print(f"[warn] 重复素材已跳过: {item['cand']['path'].name}（{KIND_CN[item['kind']]}）")
            continue
        seen.add(key)
        unique.append(item)
    selected = unique

    kinds = {item["kind"] for item in selected}
    if config.OTHER_CLASS in predictor.classes and "other" not in kinds:
        print("[warn] 没有一张被判成 other（模型仍把负类图硬判成水果）；"
              "可换宽口径 checkpoint，或检查数据/阈值")
    if "reject" not in kinds:
        print("[warn] 本轮没有拒识案例，演示将缺少'未识别为支持的水果'这一态")

    print(f"\n挑选结果 {len(selected)} 张：")
    for i, item in enumerate(selected, start=1):
        c = item["cand"]
        print(f"  {i:02d}. [{KIND_CN[item['kind']]}] {c['path'].name:<30} "
              f"预期={c['label']:<12} 预测={c['pred']:<12} {c['prob']:.3f}")
    if not (12 <= len(selected) <= 15):
        print(f"[warn] 共 {len(selected)} 张，超出建议区间 12–15；可用 --other-cases 调整")
    if args.dry_run:
        print("[dry-run] 不写任何文件")
        return

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.keep_stale:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i, item in enumerate(selected, start=1):
        c = item["cand"]
        dst = out_dir / f"{i:02d}_{item['kind']}_{c['label']}{c['path'].suffix.lower()}"
        shutil.copy2(c["path"], dst)
        items.append({
            "index": i,
            "file": rel_to_root(dst),
            "origin": rel_to_root(c["path"]),
            "source": c["source"],
            "expected": c["label"],
            "kind": item["kind"],
            "note": item["note"],
            "build": {"pred": c["pred"], "prob": round(c["prob"], 4), "state": c["state"]},
        })

    counts: Dict[str, int] = {}
    for it in items:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    save_json(Path(args.manifest), {
        "created": now_str(),
        "ckpt": rel_to_root(ckpt),
        "split": rel_to_root(split_file),
        "classes": predictor.classes,
        "temperature": round(predictor.temperature, 4),
        "threshold": None if predictor.threshold is None else round(float(predictor.threshold), 4),
        "counts": counts,
        "items": items,
    })
    print(f"\n素材目录: {rel_to_root(out_dir)}（{len(items)} 张）")
    print(f"清单: {rel_to_root(Path(args.manifest))}")
    print("下一步: python scripts/demo.py")


if __name__ == "__main__":
    main()
