#!/usr/bin/env bash
# other 负类的"单独处理"变体：把蔬果难分（番茄/黄瓜/甜椒/辣椒/茄子/柠檬）也放进 other，
# 再用这些类别里**没有被 other 占用**的图片做 OOD 测试集，验证 other 是否真的学会"这不是水果"。
#
#   在仓库根目录运行
#   PY=python bash scripts/04_broad_other_pipeline.sh
#
# 为什么：实测窄口径 other（17 个根茎类蔬菜）只有 31% 的 OOD 图被判成 other，
# 剩下 69% 被硬判成 apple/orange —— 因为番茄/柠檬这种"长得像水果"的负类根本没见过。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
SRC=${SRC:-data/raw/fruit-veg}
EPOCHS=${EPOCHS:-30}
WORKERS=${WORKERS:-8}

echo "=== 1. 11 类 + 宽口径 other（26 个蔬菜目录，含蔬果难分）==="
$PY scripts/01_data_prepare.py --src "$SRC" \
    --out data/processed_other_broad --split data/splits_other_broad.json \
    --report docs/cleaning_report_other_broad.md \
    --other-from "$SRC" --other-total 450 --other-include-ambiguous

echo "=== 2. OOD 集：同类别中未被 other 占用的图片（保证不重叠）==="
$PY scripts/02_build_ood.py --src "$SRC" --per-class 25 --clean \
    --exclude-from data/manifest_other_broad.csv

echo "=== 3. E03：11 类宽口径 other + 加权采样 ==="
$PY -m src.train --tag E03_other_broad --split data/splits_other_broad.json \
    --epochs "$EPOCHS" --num-workers "$WORKERS" --weighted-sampler \
    --note "other includes ambiguous veg; OOD held out by manifest"

echo "=== 4. 同一 OOD 集上重新标定 E01/E02/E03（口径一致才可比）==="
$PY -m src.calibrate --ckpt ckpt/E01_mbv3l_linear_best.pth --update-ckpt \
    --save-json ckpt/E01_mbv3l_linear_calib.json
$PY -m src.calibrate --ckpt ckpt/E02_other_weighted_best.pth --split data/splits_other.json \
    --update-ckpt --save-json ckpt/E02_other_weighted_calib.json
$PY -m src.calibrate --ckpt ckpt/E03_other_broad_best.pth --split data/splits_other_broad.json \
    --update-ckpt --save-json ckpt/E03_other_broad_calib.json

echo "=== 5. test 指标 + 混淆矩阵 ==="
$PY -m src.eval --ckpt ckpt/E03_other_broad_best.pth --split data/splits_other_broad.json \
    --which test --confusion-png docs/curves/cm_E03.png --save-json docs/E03_test.json

echo "=== 完成：对比 runs.csv 里 E01/E02/E03 的 test 指标与 ood_reject ==="
