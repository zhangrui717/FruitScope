"""MobileNetV3 + 三档分类头 + 冻结/BN 工具。

计划书 v2 §4.3：
- backbone 默认全程冻结；
- BN 的 running stats 必须每个 epoch 重新锁死（`model.train()` 会重置它）；
- 阶段二解冻时重建优化器（head 1e-3 / backbone 1e-4），BN 权重与 bias 不做 weight decay。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES_FALLBACK = 10


class MLPHead(nn.Module):
    """B 档：Linear→BN1d→Hardswish→Dropout→Linear（约 249k 参数）。"""

    def __init__(self, in_features: int = 960, hidden: int = 256,
                 num_classes: int = NUM_CLASSES_FALLBACK, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class OfficialHead(nn.Module):
    """C 档：官方结构 Linear(960→1280)→Hardswish→Dropout→Linear(1280→10)，约 1.24M。"""

    def __init__(self, in_features: int = 960, num_classes: int = NUM_CLASSES_FALLBACK,
                 dropout: float = 0.2, hidden: int = 1280):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def build_backbone(arch: str = "mobilenet_v3_large", pretrained: bool = True) -> nn.Module:
    """加载 ImageNet 预训练 backbone（torchvision 的权重接口，新旧版本都兼容）。"""
    weights = "DEFAULT" if pretrained else None
    if hasattr(models, "get_model"):
        return models.get_model(arch, weights=weights)
    return getattr(models, arch)(weights=weights)          # 老版本 torchvision 回退


def backbone_in_features(model: nn.Module) -> int:
    """Large=960 / Small=576；从分类头反推，换 arch 时不用手改常量。"""
    return int(model.classifier[0].in_features)


def build_model(arch: str = "mobilenet_v3_large", num_classes: int = NUM_CLASSES_FALLBACK,
                head: str = "linear", hidden: int = 256, dropout: float = 0.3,
                pretrained: bool = True) -> nn.Module:
    """按档位替换分类头：linear(A) / mlp(B) / official(C)。"""
    model = build_backbone(arch, pretrained=pretrained)
    in_features = backbone_in_features(model)
    head = head.lower()
    if head == "linear":
        model.classifier = nn.Linear(in_features, num_classes)
    elif head == "mlp":
        model.classifier = MLPHead(in_features, hidden, num_classes, dropout)
    elif head == "official":
        model.classifier = OfficialHead(in_features, num_classes, dropout)
    else:
        raise ValueError(f"未知 head 档位: {head}（可选 linear / mlp / official）")
    return model


def freeze_backbone(model: nn.Module) -> None:
    """冻结 features，只留分类头可训练。"""
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, n: int = 3) -> int:
    """解冻 features[-n:]，返回切点下标（>= cut 的 block 参与训练）。"""
    blocks = list(model.features)
    cut = max(0, len(blocks) - n)
    for blk in blocks[cut:]:
        for p in blk.parameters():
            p.requires_grad = True
    return cut


def set_frozen_bn_eval(model: nn.Module, unfreeze_last_n: int = 0) -> None:
    """把冻结 block 的 BN2d 置 eval；解冻的 block 保持 train。

    必须在每个 epoch 的 model.train() 之后调用一次，否则 BN 统计量会继续漂移。
    """
    blocks: List[nn.Module] = list(model.features)
    cut = len(blocks) - unfreeze_last_n if unfreeze_last_n > 0 else len(blocks)
    for i, blk in enumerate(blocks):
        for m in blk.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train(i >= cut)


def build_optimizer(model: nn.Module, lr_head: float = 1e-3, lr_bb: float = 1e-4,
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """四组参数：head/backbone × (衰减 / 不衰减)，BN 权重与 bias 不衰减。"""
    def split(params):
        wd, nod = [], []
        for p in params:
            if not p.requires_grad:
                continue
            (wd if p.ndim >= 2 else nod).append(p)
        return wd, nod

    h_wd, h_nod = split(model.classifier.parameters())
    b_wd, b_nod = split(model.features.parameters())
    return torch.optim.AdamW([
        {"params": h_wd, "lr": lr_head, "weight_decay": weight_decay},
        {"params": h_nod, "lr": lr_head, "weight_decay": 0.0},
        {"params": b_wd, "lr": lr_bb, "weight_decay": weight_decay},
        {"params": b_nod, "lr": lr_bb, "weight_decay": 0.0},
    ])


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int):
    """T_max 必须等于本阶段总 epoch，否则 cosine 会在中途归零。"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.classifier.parameters())
    return {"total": total, "trainable": trainable, "head": head}


def head_config(head: str, arch: str, hidden: int, dropout: float) -> dict:
    """写进 checkpoint 的头配置（推理端据此重建结构）。"""
    return {"type": head, "hidden": hidden, "dropout": dropout, "arch": arch}


def build_from_checkpoint(ckpt: dict, num_classes: int | None = None) -> nn.Module:
    """按 checkpoint 里的 arch/head 重建模型结构（不含权重）。"""
    cfg = ckpt.get("head", {})
    arch = ckpt.get("arch", cfg.get("arch", "mobilenet_v3_large"))
    n = num_classes if num_classes is not None else len(ckpt["classes"])
    return build_model(arch=arch, num_classes=n, head=cfg.get("type", "linear"),
                       hidden=cfg.get("hidden", 256), dropout=cfg.get("dropout", 0.3),
                       pretrained=False)
