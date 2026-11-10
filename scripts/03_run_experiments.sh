#!/usr/bin/env bash
# 服务器一键实验：数据准备(10类) -> 11类(含other) -> OOD集 -> 训练 -> 校准 -> test 指标
#
#   在仓库根目录运行
#   SRC=data/raw/fruit-veg bash scripts/03_run_experiments.sh
#   SRC=data/raw/fruit-veg EPOCHS=30 OTHER_TOTAL=450 bash scripts/03_run_experiments.sh
#
# 说明（对应计划书 v2 §3.4/§3.5/§4.3/§4.4）：
# - 10 类与 11 类各用一份 processed + splits，互不覆盖，指标可对比；
# - other 用 --other-total 控制总量(300–500)，默认排除番茄/黄瓜/甜椒/辣椒/茄子/柠檬；
# - 11 类那次必须 --weighted-sampler，否则 other 单类体量会主导梯度；
# - 训练后统一跑温度+阈值标定，阈值写回 checkpoint（上线直接读）。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
SRC=${SRC:-data/raw/fruit-veg}
EPOCHS=${EPOCHS:-30}
OTHER_TOTAL=${OTHER_TOTAL:-450}
WORKERS=${WORKERS:-8}

echo "=== 0. 环境自检 ==="
$PY scripts/00_env_check.py | tail -6

echo "=== 1. 数据准备：10 类（不含 other）==="
$PY scripts/01_data_prepare.py --src "$SRC" \
    --out data/processed --split data/splits.json \
    --report docs/cleaning_report.md

echo "=== 2. 数据准备：11 类（other 负类）==="
$PY scripts/01_data_prepare.py --src "$SRC" \
    --out data/processed_other --split data/splits_other.json \
    --report docs/cleaning_report_other.md \
    --other-from "$SRC" --other-total "$OTHER_TOTAL"

echo "=== 3. OOD 开集测试集（蔬果难分类别）==="
$PY scripts/02_build_ood.py --src "$SRC" --per-class 25

echo "=== 4. E01 基线：10 类，冻结 backbone ==="
$PY -m src.train --tag E01_mbv3l_linear --epochs "$EPOCHS" --num-workers "$WORKERS"

echo "=== 5. E02 开集：11 类 + other，加权采样 ==="
$PY -m src.train --tag E02_other_weighted --split data/splits_other.json \
    --epochs "$EPOCHS" --num-workers "$WORKERS" --weighted-sampler \
    --note "other negative class, weighted sampler"

echo "=== 6. 温度 + 拒识阈值标定（写回 ckpt）==="
$PY -m src.calibrate --ckpt ckpt/E01_mbv3l_linear_best.pth --update-ckpt \
    --save-json ckpt/E01_mbv3l_linear_calib.json
$PY -m src.calibrate --ckpt ckpt/E02_other_weighted_best.pth --split data/splits_other.json \
    --update-ckpt --save-json ckpt/E02_other_weighted_calib.json

echo "=== 7. test 指标 + 混淆矩阵 ==="
$PY -m src.eval --ckpt ckpt/E01_mbv3l_linear_best.pth --which test \
    --confusion-png docs/curves/cm_E01.png --save-json docs/E01_test.json
$PY -m src.eval --ckpt ckpt/E02_other_weighted_best.pth --split data/splits_other.json \
    --which test --confusion-png docs/curves/cm_E02.png --save-json docs/E02_test.json

echo "=== 完成：experiments/runs.csv / docs/cleaning_report*.md / docs/curves/ ==="
