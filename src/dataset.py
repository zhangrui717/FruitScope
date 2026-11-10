"""数据集与加载器：唯一数据入口是 data/splits.json。

这样 D4–D6 所有实验读同一个划分文件，指标才可比较（计划书 v2 §3.4、§3.5）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from src.transforms import build_eval_transform, build_train_transform, load_image
from src.utils import load_json, worker_init_fn


def resolve_path(path_like: str | Path) -> Path:
    """splits.json 里存的是相对项目根的路径（如 data/processed/apple/1.jpg）。"""
    p = Path(path_like)
    return p if p.is_absolute() else (config.ROOT / p)


def load_split_file(split_file: str | Path = config.SPLIT_FILE) -> Dict:
    """读取划分文件并做基本校验（缺字段直接报错，别让它静默跑偏）。"""
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(
            f"找不到划分文件 {split_file}，先跑: python scripts/01_data_prepare.py --help")
    payload = load_json(split_file)
    for key in ("classes", "splits"):
        if key not in payload:
            raise KeyError(f"{split_file} 缺少字段 {key}")
    for name in ("train", "val", "test"):
        if name not in payload["splits"]:
            raise KeyError(f"{split_file} 缺少 {name} 划分")
    return payload


class FruitDataset(Dataset):
    """按 splits.json 的 records 逐条读图；类别索引来自 classes 列表顺序。"""

    def __init__(self, records: Sequence[Dict[str, str]], classes: Sequence[str], transform=None):
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        missing = {r["label"] for r in records} - set(self.classes)
        if missing:
            raise ValueError(f"records 中出现未登记的类别: {sorted(missing)}")
        self.samples: List[Tuple[Path, int]] = [
            (resolve_path(r["path"]), self.class_to_idx[r["label"]]) for r in records
        ]
        self.transform = transform or build_eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            img = load_image(path)
        except Exception as exc:                       # 坏图要报出具体路径，别只说 "OSError"
            raise RuntimeError(f"读取图片失败: {path} ({exc})") from exc
        return self.transform(img), label

    def class_counts(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


def build_datasets(split_file: str | Path = config.SPLIT_FILE,
                   img_size: int = config.IMG_SIZE) -> Dict[str, FruitDataset]:
    """按划分文件构造 train/val/test 三个数据集（transform 各用各的）。"""
    payload = load_split_file(split_file)
    classes = payload["classes"]
    train_tf = build_train_transform(img_size)
    eval_tf = build_eval_transform(img_size)
    datasets: Dict[str, FruitDataset] = {}
    for name in ("train", "val", "test"):
        tf = train_tf if name == "train" else eval_tf
        datasets[name] = FruitDataset(payload["splits"][name], classes, transform=tf)
    datasets["classes"] = classes        # type: ignore[assignment]
    return datasets


def inverse_frequency_weights(dataset: FruitDataset) -> np.ndarray:
    """类别越少权重越高（混入 Fruits-360 后某类暴涨时用）。"""
    labels = np.array([label for _, label in dataset.samples])
    counts = np.bincount(labels, minlength=len(dataset.classes)).astype(np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    return weights[labels]


def build_loader(dataset: FruitDataset, batch_size: int = config.DEFAULTS["batch_size"],
                 shuffle: bool = False, num_workers: int = config.DEFAULTS["num_workers"],
                 weighted: bool = False, drop_last: bool | None = None,
                 seed: int = config.SEED) -> DataLoader:
    """drop_last 默认只对训练集开（防 BatchNorm1d 在最后一批 batch=1 时崩）。"""
    if drop_last is None:
        drop_last = shuffle
    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    if weighted:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(inverse_frequency_weights(dataset), dtype=torch.double),
            num_samples=len(dataset), replacement=True, generator=generator)
        shuffle = False

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=drop_last, generator=generator, worker_init_fn=worker_init_fn,
    )


def build_all_loaders(split_file: str | Path = config.SPLIT_FILE,
                      batch_size: int = config.DEFAULTS["batch_size"],
                      num_workers: int = config.DEFAULTS["num_workers"],
                      weighted: bool = False, seed: int = config.SEED
                      ) -> Tuple[Dict[str, DataLoader], List[str], Dict[str, int]]:
    """返回 (loaders, classes, 各划分样本数)。"""
    datasets = build_datasets(split_file)
    classes: List[str] = datasets.pop("classes")     # type: ignore[assignment]
    loaders = {
        "train": build_loader(datasets["train"], batch_size, shuffle=True,
                              num_workers=num_workers, weighted=weighted, seed=seed),
        "val": build_loader(datasets["val"], batch_size, shuffle=False, num_workers=num_workers),
        "test": build_loader(datasets["test"], batch_size, shuffle=False, num_workers=num_workers),
    }
    sizes = {k: len(datasets[k]) for k in ("train", "val", "test")}   # type: ignore[index]
    return loaders, classes, sizes
