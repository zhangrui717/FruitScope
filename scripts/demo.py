"""终端演示：逐张播放 `demo/manifest.json` 里的固定图片，打印三态文案并留存标注图。

完全复用 `src/predict.py::Predictor`（同一个预处理、同一个温度校准、同一个拒识阈值），
因此**演示结果与正式评估一致**，不存在"演示专用推理逻辑"。

    python scripts/demo.py                          # 默认 E04，交互式播放（每张停 1.5s）
    python scripts/demo.py --interval 2.5            # 放慢
    python scripts/demo.py --no-wait --no-color      # 静默（管道/重定向/后台自动如此）
    python scripts/demo.py --ckpt ckpt/E01_mbv3l_linear_best.pth   # 换 10 类模型

非 TTY（`| tee`、`> log`、nohup）时自动关闭等待与颜色，重定向也不会出现 ANSI 乱码。
标注图写到 `demo/output/`（matplotlib Agg 后端，不会弹窗）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.predict import Predictor
from src.transforms import load_image
from src.utils import load_json, now_str, rel_to_root, save_json

KIND_CN = {
    "correct": "正确命中",
    "other": "非目标(other)",
    "reject": "低置信拒识",
    "misclass": "边界误判",
}
STATE_CN = {"fruit": "识别水果", "other": "非目标(other)", "reject": "拒识"}


# ---------------- 终端能力 ----------------

def enable_ansi() -> bool:
    """Windows 10+ 控制台默认不开 VT，这里显式打开；失败就退回无色。"""
    if os.name != "nt":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def unicode_ok() -> bool:
    """cp936 之类编码下 `█ ✓ →` 会直接抛 UnicodeEncodeError，探测一次再决定字符集。"""
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    try:
        "█·✓✗→".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class Style:
    """极简 ANSI 着色；enabled=False 时原样返回（管道里就是纯文本）。"""

    CODES = {"green": "32", "red": "31", "yellow": "33", "cyan": "36",
             "magenta": "35", "dim": "2", "bold": "1"}

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        codes = ";".join(self.CODES[n] for n in names if n in self.CODES)
        return f"\x1b[{codes}m{text}\x1b[0m"


def emit(text: str = "") -> None:
    """任何编码下都不崩的打印（终端编码太窄就替换掉不可编码字符）。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii", "replace").decode(
            sys.stdout.encoding or "ascii", "replace"))


def bar(prob: float, width: int = 26, fancy: bool = True) -> str:
    p = max(0.0, min(1.0, float(prob)))
    filled = int(round(p * width))
    full, rest = ("█", "·") if fancy else ("#", "-")
    return full * filled + rest * (width - filled)


def state_of(result: dict) -> str:
    """三态：拒识 / other / 正常水果（与 predict.py 的文案分支一一对应）。"""
    if result["rejected"]:
        return "reject"
    if result["topk"][0]["label"] == config.OTHER_CLASS:
        return "other"
    return "fruit"


def verdict(record: dict, result: dict) -> Tuple[bool, str]:
    """这条素材是否复现了 manifest 里写的预期。"""
    kind = record["kind"]
    pred = result["topk"][0]["label"]
    state = state_of(result)
    if kind == "reject":
        return state == "reject", "预期被拒识"
    if kind == "other":
        return state == "other", "预期判成 other"
    if kind == "correct":
        return (state == "fruit" and pred == record["expected"]), "预期判对"
    return (state == "fruit" and pred != record["expected"]), "预期仍判错（诚实展示局限）"


# ---------------- 标注图（Agg，不弹窗） ----------------

CJK_FONTS = ("Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "Source Han Sans CN",
             "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei",
             "PingFang SC", "Heiti SC", "Arial Unicode MS", "Noto Sans CJK JP")


def find_cjk_font() -> Optional[str]:
    """服务器常见只有 DejaVu Sans（没有中文字形），找不到就退回英文标注。"""
    try:
        from matplotlib import font_manager
    except Exception:
        return None
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in CJK_FONTS:
        if cand in names:
            return cand
    return None


def save_annotated(record: dict, result: dict, out_dir: Path,
                   cjk_font: Optional[str], index: int, total: int) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")                 # 必须在 import pyplot 之前
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
    except Exception as exc:                  # matplotlib 缺失不该让演示挂掉
        print(f"[warn] 跳过标注图（matplotlib 不可用: {exc}）")
        return None

    if cjk_font:
        rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    # 中文字体常常没有 bold/normal 两个字重，matplotlib 会逐条打 findfont 警告，
    # 混进 nohup/重定向日志里很难看，这里只保留错误级别。
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    cn = cjk_font is not None
    img = load_image(record["_path"])
    w, h = img.size
    fig_w, img_h, text_h = 5.6, max(2.4, min(5.6 * h / max(w, 1), 4.8)), 2.0
    fig = plt.figure(figsize=(fig_w, img_h + text_h), dpi=110)

    ax_img = fig.add_axes([0.02, (text_h + 0.25) / (img_h + text_h), 0.96,
                           img_h / (img_h + text_h) - 0.02])
    ax_img.imshow(img)
    ax_img.axis("off")

    ok, hint = verdict(record, result)
    state = state_of(result)
    pred = result["topk"][0]
    state_color = {"fruit": "#1a7f37", "other": "#b8860b", "reject": "#c0392b"}[state]
    exp_cn = config.class_cn(record["expected"]) if cn else record["expected"]
    pred_name = pred["name_cn"] if cn else pred["label"]
    top3 = "  ".join(f"{t['name_cn'] if cn else t['label']} {t['prob']:.1%}"
                     for t in result["topk"])

    if cn:
        lines = [
            (f"[{index}/{total}] {Path(record['file']).name}", 10.5, "#333333", "normal"),
            (f"预期 {exp_cn}  →  预测 {pred_name}（{pred['prob']:.1%}）", 12, "#111111", "bold"),
            (f"三态：{STATE_CN[state]}    {'✓ ' + hint if ok else '✗ 未复现预期'}", 10.5,
             state_color, "bold"),
            (f"文案：{result['message']}", 10, "#444444", "normal"),
            (f"Top3：{top3}", 9.5, "#666666", "normal"),
        ]
    else:
        lines = [
            (f"[{index}/{total}] {Path(record['file']).name}", 10.5, "#333333", "normal"),
            (f"expected {record['expected']}  ->  predicted {pred['label']} ({pred['prob']:.1%})",
             12, "#111111", "bold"),
            (f"state: {state}    {'OK' if ok else 'MISMATCH'}", 10.5, state_color, "bold"),
            (f"message: {result['message']}", 9, "#444444", "normal"),
            (f"Top3: {top3}", 9.5, "#666666", "normal"),
        ]

    ax_txt = fig.add_axes([0.02, 0.01, 0.96, text_h / (img_h + text_h) - 0.03])
    ax_txt.axis("off")
    y = 0.94
    for text, size, color, weight in lines:
        ax_txt.text(0.0, y, text, fontsize=size, color=color, weight=weight,
                    transform=ax_txt.transAxes, va="top", wrap=True)
        y -= 0.20 if size >= 12 else 0.175

    out = out_dir / f"{index:02d}_{record['kind']}_{record['expected']}.png"
    try:
        fig.savefig(out, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)                        # 绝不 plt.show()
    return out


# ---------------- 终端渲染 ----------------

def _dw(text: str) -> int:
    """终端显示宽度：中文占两列，靠它对齐。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _dw(text))


def _rule(width: int = 76, fancy: bool = True) -> str:
    return ("─" if fancy else "-") * width


def render(index: int, total: int, record: dict, result: dict, style: Style,
           fancy: bool) -> None:
    ok, hint = verdict(record, result)
    state = state_of(result)
    pred = result["topk"][0]
    exp = record["expected"]
    exp_cn = config.class_cn(exp)

    mark = ("✓" if fancy else "OK") if ok else ("✗" if fancy else "NO")
    mark_color = "green" if ok else "red"
    state_color = {"fruit": "green", "other": "yellow", "reject": "red"}[state]

    emit(style(_rule(76, fancy), "dim"))
    emit(style(f"[{index:02d}/{total:02d}]", "bold", "cyan") + " " +
         style(Path(record["file"]).name, "bold") + "   " +
         style(f"预期 {exp_cn}({exp}) · {KIND_CN.get(record['kind'], record['kind'])}", "dim"))
    emit("  预测   " + style(f"{pred['name_cn']}({pred['label']})", "bold") +
         f"  {pred['prob']:.3f}" + "   " +
         style(f"{mark} {hint}", mark_color, "bold"))
    for i, item in enumerate(result["topk"]):
        label = _pad(f"{item['name_cn']}({item['label']})", 22)
        emit(("  Top3   " if i == 0 else "         ") + label +
             f"{item['prob']:.3f}  " + style(bar(item["prob"], 26, fancy), "cyan"))
    thr = ""
    if state == "reject" and result.get("threshold") is not None:
        thr = f"（阈值 {float(result['threshold']):.3f}）"
    emit("  三态   " + style(STATE_CN[state], state_color, "bold") + style(thr, "dim"))
    emit("  文案   " + style(result["message"], "bold"))
    emit("  说明   " + style(record.get("note", ""), "dim"))


def print_summary(rows: List[dict], style: Style, fancy: bool, manifest: dict,
                  ckpt_name: str) -> Dict[str, object]:
    kinds = Counter(r["kind"] for r in rows)
    states = Counter(r["state"] for r in rows)
    matched = sum(1 for r in rows if r["ok"])
    fruit_rows = [r for r in rows if r["kind"] in ("correct", "misclass")]
    fruit_hit = sum(1 for r in fruit_rows if r["pred"] == r["expected"])
    acc = fruit_hit / len(fruit_rows) if fruit_rows else float("nan")

    emit("")
    emit(style(_rule(76, fancy), "dim"))
    emit(style("汇总", "bold", "cyan") + f"  {len(rows)} 张素材 · checkpoint {ckpt_name}")
    emit("  预期类型  " + "   ".join(
        f"{KIND_CN[k]} {kinds.get(k, 0)}" for k in ("correct", "other", "reject", "misclass")))
    emit("  三态次数  " + "   ".join(
        style(f"{STATE_CN[k]} {states.get(k, 0)}",
              {"fruit": "green", "other": "yellow", "reject": "red"}[k])
        for k in ("fruit", "other", "reject")))
    emit(f"  复现预期  {matched}/{len(rows)}（{matched / len(rows):.1%}）")
    if fruit_rows:
        emit(f"  水果类准确率  {fruit_hit}/{len(fruit_rows)} = {acc:.1%}"
             + style("   （只算 manifest 里预期为水果的那些）", "dim"))
    bad = [r for r in rows if not r["ok"]]
    if bad:
        emit(style("  未复现预期  ", "red") + ", ".join(
            f"{Path(r['file']).name}（预期 {KIND_CN[r['kind']]}，实际 {STATE_CN[r['state']]}）"
            for r in bad))
    else:
        emit(style("  全部素材都复现了预期 ✓", "green"))
    return {
        "kinds": dict(kinds),
        "states": dict(states),
        "matched": matched,
        "fruit_accuracy": None if not fruit_rows else round(acc, 4),
        "manifest_created": manifest.get("created"),
        "ckpt": ckpt_name,
    }


# ---------------- 主流程 ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="FruitScope 终端演示（逐张播放固定素材）")
    ap.add_argument("--ckpt", default=str(config.DEMO_DEFAULT_CKPT),
                    help="默认 E04 开集版；换 10 类可用 ckpt/E01_mbv3l_linear_best.pth")
    ap.add_argument("--manifest", default=str(config.DEMO_MANIFEST))
    ap.add_argument("--interval", type=float, default=config.DEMO_INTERVAL,
                    help="每张停留秒数（非 TTY 时自动不等待）")
    ap.add_argument("--no-wait", action="store_true", help="不等待，一口气跑完")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default=str(config.DEMO_OUTPUT_DIR))
    ap.add_argument("--no-save", action="store_true", help="不生成标注图")
    ap.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")
    ap.add_argument("--device", default=None, help="cpu / cuda（默认自动）")
    ap.add_argument("--strict", action="store_true", help="有素材未复现预期就以 1 退出")
    args = ap.parse_args()

    interactive = sys.stdout.isatty()
    color_on = (interactive and not args.no_color
                and not os.environ.get("NO_COLOR") and enable_ansi())
    style = Style(bool(color_on))
    fancy = unicode_ok()
    wait = bool((not args.no_wait) and interactive and args.interval > 0)

    # ---- 1) 清单 ----
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(
            f"找不到演示清单: {manifest_path}\n"
            "先生成素材（从测试集/OOD 集挑 12–15 张固定图片）：\n"
            "  python scripts/05_build_demo.py\n"
            "  # 或指定：--ckpt ckpt/E04_broad_unfreeze3_best.pth "
            "--split data/splits_other_broad.json")
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        raise SystemExit(f"演示清单无法解析（JSON 损坏？）: {manifest_path}\n{exc}")

    records = manifest.get("items") or []
    if not records:
        raise SystemExit(f"演示清单为空: {manifest_path}\n重新生成: python scripts/05_build_demo.py")

    missing = []
    for rec in records:
        p = Path(rec["file"])
        rec["_path"] = p if p.is_absolute() else config.ROOT / p
        if not rec["_path"].exists():
            missing.append(str(rec["file"]))
    if missing:
        shown = "\n  ".join(missing[:8])
        if len(missing) > 8:
            shown += f"\n  ...（共 {len(missing)} 张）"
        raise SystemExit(f"演示图片缺失：\n  {shown}\n"
                         "重新生成素材: python scripts/05_build_demo.py")

    # ---- 2) checkpoint（复用 src.predict.Predictor，绝不另写推理） ----
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        available = sorted(p.name for p in config.CKPT_DIR.glob("*.pth"))
        raise SystemExit(
            f"找不到 checkpoint: {ckpt}\n"
            f"ckpt/ 下现有: {available or '空'}\n"
            "--ckpt ckpt/E01_mbv3l_linear_best.pth      # 10 类\n"
            "--ckpt ckpt/E04_broad_unfreeze3_best.pth   # 11 类开集（默认）")
    try:
        predictor = Predictor(ckpt, topk=args.topk, device=args.device)
    except Exception as exc:
        raise SystemExit(f"checkpoint 加载失败: {ckpt}\n{type(exc).__name__}: {exc}")

    thr = "None" if predictor.threshold is None else f"{float(predictor.threshold):.3f}"
    emit(style("FruitScope 演示", "bold", "cyan") +
         style(f"  {len(predictor.classes)} 类 · {len(records)} 张素材", "dim"))
    emit(f"  checkpoint  {rel_to_root(ckpt)}")
    emit(f"  校准温度    T={predictor.temperature:.3f}    拒识阈值 {thr}    设备 {predictor.device}")
    mode = (f"交互播放（每张 {args.interval:.1f}s）" if wait
            else "静默模式（不等待/无颜色）" if not interactive else "无等待")
    emit(style(f"  运行模式    {mode}", "dim"))
    emit("")

    # ---- 3) 逐张播放 ----
    out_dir = Path(args.out)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
    cjk_font = find_cjk_font() if not args.no_save else None
    if not args.no_save and cjk_font is None:
        emit(style("[note] 没找到中文字体，标注图用英文标注（可装 fonts-noto-cjk 改善）", "dim"))

    rows: List[dict] = []
    t0 = time.time()
    try:
        for i, rec in enumerate(records, start=1):
            try:
                result = predictor.predict(rec["_path"])
            except Exception as exc:              # 单张坏图跳过，不让整批中断
                emit(style(f"[{i:02d}/{len(records):02d}] 读取失败，跳过 "
                           f"{rec['_path'].name}: {exc}", "red"))
                continue
            ok, _ = verdict(rec, result)
            rows.append({"index": i, "file": rec["file"], "kind": rec["kind"],
                         "expected": rec["expected"], "pred": result["topk"][0]["label"],
                         "prob": round(float(result["confidence"]), 4),
                         "state": state_of(result), "ok": ok,
                         "message": result["message"]})
            render(i, len(records), rec, result, style, fancy)
            if not args.no_save:
                fig = save_annotated(rec, result, out_dir, cjk_font, i, len(records))
                if fig is not None:
                    emit(style(f"  标注图   {rel_to_root(fig)}", "dim"))
            if wait and i < len(records):
                time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        emit("")
        emit(style("已中断（Ctrl-C），只汇总已跑完的部分", "yellow"))

    if not rows:
        raise SystemExit("一张图都没跑成功，请检查图片与 checkpoint")
    elapsed = time.time() - t0

    # ---- 4) 汇总 + 留档 ----
    summary = print_summary(rows, style, fancy, manifest, ckpt.name)
    emit(style(f"  总耗时      {elapsed:.1f}s（{elapsed / len(rows) * 1000:.0f} ms/张）", "dim"))
    if not args.no_save:
        save_json(out_dir / "summary.json", {
            "created": now_str(),
            "ckpt": rel_to_root(ckpt),
            "manifest": rel_to_root(manifest_path),
            "device": str(predictor.device),
            "temperature": round(predictor.temperature, 4),
            "threshold": None if predictor.threshold is None else round(float(predictor.threshold), 4),
            "elapsed_sec": round(elapsed, 2),
            "summary": summary,
            "records": rows,
        })
        emit(style(f"  标注图/汇总 {rel_to_root(out_dir)}/  "
                   f"（拉回本机: scp -r <host>:{rel_to_root(out_dir)} .）", "dim"))

    if args.strict and any(not r["ok"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
