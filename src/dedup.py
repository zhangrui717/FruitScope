"""感知哈希（pHash）去重：纯 numpy 实现，不依赖 imagehash。

计划书 v2 §3.5：同源图片常同时出现在 train/val/test，造成虚假精度，
因此入库前做 pHash 去重，划分后再做一次跨划分泄漏检查。

近邻检索用 banding（鸽笼原理）：把 64 bit 切成 threshold+1 段，
汉明距离 <= threshold 的两张图至少有一段完全相同，只在同段候选里比对。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps


def _dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵（n x n），用于 32x32 的二维 DCT。"""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    mat = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    mat[0, :] *= np.sqrt(1.0 / n)
    mat[1:, :] *= np.sqrt(2.0 / n)
    return mat


_DCT32 = _dct_matrix(32)


def phash(path: str | Path, hash_size: int = 8) -> int:
    """返回 64 bit 感知哈希（int）。先转灰度、缩放 32x32、取 DCT 左上低频块再取中位数。"""
    size = hash_size * 4
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size, size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32)
    dct = _DCT32 @ arr @ _DCT32.T
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, :])            # 去掉直流分量再取中位数
    bits = (low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def _band_keys(value: int, n_bands: int) -> List[str]:
    """把 64 bit 切成 n_bands 段，返回每段的 (段号, 段内容) 字符串键。"""
    bits = format(value, "064b")
    edges = np.linspace(0, 64, n_bands + 1).astype(int)
    return [f"{i}:{bits[edges[i]:edges[i + 1]]}" for i in range(n_bands)]


def find_duplicate_pairs(hashes: Dict[str, int], threshold: int = 8
                         ) -> List[Tuple[str, str, int]]:
    """返回所有汉明距离 <= threshold 的相似对（key 一般为相对路径）。"""
    buckets: Dict[str, List[str]] = {}
    pairs: List[Tuple[str, str, int]] = []
    n_bands = max(2, threshold + 1)
    seen: set[Tuple[str, str]] = set()

    for key, value in hashes.items():
        for band in _band_keys(value, n_bands):
            for other in buckets.get(band, ()):            # 只看同段候选
                if other == key:
                    continue
                pair = (other, key) if other < key else (key, other)
                if pair in seen:
                    continue
                dist = hamming(hashes[pair[0]], hashes[pair[1]])
                if dist <= threshold:
                    seen.add(pair)
                    pairs.append((pair[0], pair[1], dist))
        for band in _band_keys(value, n_bands):
            buckets.setdefault(band, []).append(key)
    return pairs


def dedupe_by_phash(items: Sequence[Tuple[str, Path]], threshold: int = 8
                    ) -> Tuple[List[Tuple[str, Path]], List[Dict[str, object]]]:
    """同类内去重：按顺序保留先出现的，后出现的近重复丢弃。

    items: [(key, path)]，返回 (保留项, 丢弃记录)。
    """
    kept: List[Tuple[str, Path]] = []
    kept_hashes: List[int] = []
    dropped: List[Dict[str, object]] = []
    n_bands = max(2, threshold + 1)
    buckets: Dict[str, List[int]] = {}

    for key, path in items:
        value = phash(path)
        candidates = set()
        for band in _band_keys(value, n_bands):
            candidates.update(buckets.get(band, ()))
        hit = None
        for idx in candidates:
            if hamming(value, kept_hashes[idx]) <= threshold:
                hit = idx
                break
        if hit is None:
            idx = len(kept_hashes)
            kept.append((key, path))
            kept_hashes.append(value)
            for band in _band_keys(value, n_bands):
                buckets.setdefault(band, []).append(idx)
        else:
            dropped.append({"kept": kept[hit][0], "dropped": key,
                            "distance": hamming(value, kept_hashes[hit])})
    return kept, dropped


def cross_split_leaks(splits: Dict[str, Iterable[str]], hashes: Dict[str, int],
                      threshold: int = 8, max_report: int = 20) -> List[Dict[str, object]]:
    """划分完成后检查：val/test 里是否有与 train 近重复的图（会虚高指标）。"""
    name_by_path = {p: name for name, paths in splits.items() for p in paths}
    leaks: List[Dict[str, object]] = []
    train_paths = [p for p in splits.get("train", []) if p in hashes]

    index: Dict[str, List[str]] = {}
    n_bands = max(2, threshold + 1)
    for p in train_paths:
        for band in _band_keys(hashes[p], n_bands):
            index.setdefault(band, []).append(p)

    for name in ("val", "test"):
        for p in splits.get(name, []):
            if p not in hashes:
                continue
            hit = None
            for band in _band_keys(hashes[p], n_bands):
                for other in index.get(band, ()):
                    if hamming(hashes[p], hashes[other]) <= threshold:
                        hit = other
                        break
                if hit:
                    break
            if hit and len(leaks) < max_report * 50:
                leaks.append({"split": name, "image": p, "train_match": hit,
                              "distance": hamming(hashes[p], hashes[hit])})
    return leaks
