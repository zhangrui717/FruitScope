"""上线推理参考实现：top-k + 温度校准概率 + 拒识判断。

计划书 v2 §4.5：checkpoint 自带 classes / img_size / mean / std / head / 温度 / 阈值，
推理端不得另写一套预处理。

    python -m src.predict --image D:/photos/a.jpg            # 单图
    python -m src.predict --dir D:/photos --topk 3           # 批量
    python -m src.predict --selfcheck                        # 用 val 前 N 张自检
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.model import build_from_checkpoint
from src.transforms import build_eval_transform, load_image, tensor_from_path
from src.utils import iter_files, load_json, set_seed, softmax_np


class Predictor:
    """把 checkpoint 里的全部推理约定收在一处。"""

    def __init__(self, ckpt_path: str | Path = config.MODEL_DIR / "fruitscope.pth",
                 topk: int = 3, device: str | None = None):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.classes = list(ckpt["classes"])
        self.temperature = float(ckpt.get("temperature") or 1.0)
        self.threshold = ckpt.get("threshold")
        self.img_size = int(ckpt.get("img_size", config.IMG_SIZE))
        self.topk = int(min(topk, len(self.classes)))
        self.ckpt_meta = {k: ckpt.get(k) for k in ("arch", "head", "env", "val", "test")}

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_from_checkpoint(ckpt)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_eval_transform(
            self.img_size, mean=ckpt.get("mean", config.MEAN), std=ckpt.get("std", config.STD)
        )

    def predict(self, image_path: str | Path | BinaryIO) -> dict:
        """返回 top-k 与拒识结论（threshold 为 None 表示未标定，一律不拒）。"""
        x = tensor_from_path(image_path, self.transform).to(self.device)
        with torch.no_grad():
            logits = self.model(x).float().cpu().numpy()
        probs = softmax_np(logits / max(self.temperature, 1e-6))[0]
        order = np.argsort(-probs)[: self.topk]
        top_label = self.classes[int(order[0])]
        rejected = bool(self.threshold is not None and float(probs.max()) < float(self.threshold))
        return {
            "image": str(image_path) if isinstance(image_path, (str, Path)) else "<image stream>",
            "topk": [{"label": self.classes[i], "name_cn": config.class_cn(self.classes[i]),
                      "prob": float(probs[i])} for i in order],
            "confidence": float(probs.max()),
            "rejected": rejected,
            "threshold": self.threshold,
            "message": self._message(top_label, rejected),
        }

    @staticmethod
    def _message(top_label: str, rejected: bool) -> str:
        """上线文案：拒识 / other 负类 / 正常命中，三态分开（计划书 §4.4 配套项）。"""
        if rejected:
            return "未识别为支持的水果，请换一张更清晰、主体更完整的照片"
        if top_label == config.OTHER_CLASS:
            return "这不是我认识的水果（支持苹果/香蕉/葡萄/猕猴桃/芒果/橙子/梨/菠萝/石榴/西瓜）"
        return f"可能是{config.class_cn(top_label)}"


def selfcheck(ckpt_path: str | Path, split_file: str | Path = config.SPLIT_FILE, n: int = 5) -> int:
    """用拆分文件里的 val 前 n 张跑一遍，验证"预处理 + 类别顺序"没接错。"""
    predictor = Predictor(ckpt_path)
    splits = load_json(split_file)["splits"]["val"][:n]
    fails = 0
    for record in splits:
        result = predictor.predict(config.ROOT / record["path"])
        top1 = result["topk"][0]["label"]
        ok = top1 == record["label"]
        fails += int(not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {Path(record['path']).name}: "
              f"gt={record['label']} pred={top1} p={result['confidence']:.3f}")
    print(f"selfcheck: {len(splits) - fails}/{len(splits)} 命中")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description="FruitScope 推理")
    ap.add_argument("--ckpt", default=str(config.MODEL_DIR / "fruitscope.pth"))
    ap.add_argument("--split", default=str(config.SPLIT_FILE))
    ap.add_argument("--image", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--strict", action="store_true", help="selfcheck 有错就返回非 0")
    args = ap.parse_args()

    set_seed(config.SEED)
    if args.selfcheck:
        fails = selfcheck(args.ckpt, args.split)
        raise SystemExit(1 if (fails and args.strict) else 0)

    predictor = Predictor(args.ckpt, topk=args.topk)
    paths = [Path(args.image)] if args.image else (list(iter_files(Path(args.dir))) if args.dir else [])
    if not paths:
        ap.error("需要 --image / --dir / --selfcheck 之一")
    for p in paths:
        out = predictor.predict(p)
        top = " | ".join(f"{t['name_cn']}({t['label']}) {t['prob']:.3f}" for t in out["topk"])
        flag = "拒识" if out["rejected"] else "识别"
        print(f"{p.name}: {flag} -> {top}  [{out['message']}]")


if __name__ == "__main__":
    main()
