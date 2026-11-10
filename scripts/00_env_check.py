"""环境自检：输出 docs/env.md 与 docs/env_lock.txt（计划书 v2 §4.2 D1 / §4.5）。

    python scripts/00_env_check.py                     # 只查环境
    python scripts/00_env_check.py --split data/splits.json   # 顺带核对数据与划分
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.utils import now_str


def collect(split_file: Path | None) -> dict:
    info = {
        "time": now_str(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        import torchvision
        info["torch"] = torch.__version__
        info["torchvision"] = torchvision.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cudnn"] = torch.backends.cudnn.version()
    except ModuleNotFoundError as exc:
        info["torch_error"] = str(exc)

    total, used, free = shutil.disk_usage(config.ROOT)
    info["disk_free_gb"] = round(free / 2 ** 30, 1)

    if split_file and Path(split_file).exists():
        from src.utils import load_json
        payload = load_json(Path(split_file))
        info["split"] = {
            "file": str(split_file),
            "seed": payload.get("seed"),
            "classes": payload.get("classes"),
            "sizes": {k: len(v) for k, v in payload["splits"].items()},
            "dedupe": payload.get("dedupe"),
        }
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="环境自检")
    ap.add_argument("--split", default=str(config.SPLIT_FILE))
    args = ap.parse_args()

    info = collect(Path(args.split))
    config.ensure_dirs()

    md = ["# 运行环境", "", f"- 采集时间：{info['time']}", "",
          "| 项 | 值 |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in info.items() if k != "split"]
    if "split" in info:
        s = info["split"]
        md += ["", "## 数据划分", "",
               f"- 文件：`{s['file']}`（seed={s['seed']}）",
               f"- 类别（{len(s['classes'])}）：{', '.join(s['classes'])}",
               f"- 张数：{s['sizes']}",
               f"- 去重：{s['dedupe']}"]
    md += ["", "## 复现提示", "",
           "- 服务器上先激活自己的训练环境（原实验环境使用 "
           "torch 2.11.0+cu128，4×RTX 3090）。",
           "- 依赖清单见 `docs/env_lock.txt`（pip freeze）。",
           "- 训练脚本写入的 checkpoint 里也带 env 字段，换机器前先比对。"]

    env_md = config.DOCS_DIR / "env.md"
    env_md.write_text("\n".join(md), encoding="utf-8")
    print(f"环境报告: {env_md}")

    lock = config.DOCS_DIR / "env_lock.txt"
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120)
        lock.write_text(out.stdout, encoding="utf-8")
        print(f"依赖锁定: {lock}（{len(out.stdout.splitlines())} 条）")
    except Exception as exc:
        print(f"[warn] pip freeze 失败：{exc}")

    print("\n".join(f"{k}: {v}" for k, v in info.items() if k != "split"))


if __name__ == "__main__":
    main()
