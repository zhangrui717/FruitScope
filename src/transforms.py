"""图像预处理：训练增强 / 评估预处理 / 上线推理共用同一套。

计划书 v2 §3.4（增强修正）+ §4.5（预处理三件套）。
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps
from torchvision import transforms as T

import config


def build_train_transform(img_size: int = config.IMG_SIZE) -> T.Compose:
    """训练增强：顺序即执行顺序，RandomErasing 必须在 Normalize 之后。"""
    return T.Compose([
        # ① 限定 scale：默认 (0.08,1.0) 会把单主体水果裁成"一片果皮/纯背景"
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        # ② 左右翻转对水果安全；不要上下翻转（水果有天然朝向）
        T.RandomHorizontalFlip(p=0.5),
        # ③ 用数据集中位色填充，避免纯黑边角引入训练/推理都不存在的伪域
        T.RandomAffine(degrees=15, translate=(0.05, 0.05), fill=config.FILL),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(config.MEAN, config.STD),
        # ④ 作用在 tensor 上，p 从 0.5 降到 0.25
        T.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def build_eval_transform(
    img_size: int = config.IMG_SIZE, *, mean=config.MEAN, std=config.STD
) -> T.Compose:
    """验证 / 测试 / 上线推理共用（禁止上线端另写一套）。"""
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),   # 短边缩放并保持长宽比（不是 Resize((224,224))）
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_image(path: str | Path | BinaryIO, pad_fill=config.FILL, max_aspect: float = config.MAX_ASPECT) -> Image.Image:
    """读取图片并做上线必做的三件事：EXIF 摆正、转 RGB、极端长宽比先 pad 成方形。"""
    with Image.open(path) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")

    w, h = img.size
    if min(w, h) == 0:
        raise ValueError(f"图片尺寸非法: {path}")
    if max(w, h) / min(w, h) > max_aspect:  # 长截图/全景：直接 CenterCrop 会把主体裁掉
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), pad_fill)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    return img


def tensor_from_path(path: str | Path | BinaryIO, transform=None):
    """路径 -> (1,3,H,W) tensor，供推理使用。"""
    tf = transform or build_eval_transform()
    return tf(load_image(path)).unsqueeze(0)
