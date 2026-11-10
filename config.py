"""FruitScope 全局配置。

训练 / 评估 / 校准 / 推理统一从这里取路径与常量，避免"上线端另写一套"
导致的像素级预处理或类别顺序不一致。对应计划书 v2 §3.4 / §4.3 / §4.5。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- 目录 ----------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"          # 扁平结构：data/processed/<class>/<img>
SPLIT_FILE = DATA_DIR / "splits.json"           # 唯一真相（进 Git）
FRUITS360_MAP_FILE = DATA_DIR / "fruits360_map.json"
REAL_HOLDOUT_DIR = DATA_DIR / "real_holdout"    # 自拍独立集，只做展示/坏案例
OOD_DIR = DATA_DIR / "ood"                      # 番茄/柠檬/非水果等负样本

CKPT_DIR = ROOT / "ckpt"
MODEL_DIR = ROOT / "model"
LOG_DIR = ROOT / "logs"
EXP_DIR = ROOT / "experiments"
RUNS_CSV = EXP_DIR / "runs.csv"
DOCS_DIR = ROOT / "docs"

# ---------- 终端演示（scripts/demo.py） ----------
DEMO_DIR = ROOT / "demo"
DEMO_SAMPLES_DIR = DEMO_DIR / "samples"     # 固定副本（12–15 张），进 Git
DEMO_OUTPUT_DIR = DEMO_DIR / "output"       # 标注图 + summary.json，不进 Git
DEMO_MANIFEST = DEMO_DIR / "manifest.json"

# 其他数据集划分（与 SPLIT_FILE 并列；11 类开集用这两个）
SPLIT_OTHER_FILE = DATA_DIR / "splits_other.json"
SPLIT_OTHER_BROAD_FILE = DATA_DIR / "splits_other_broad.json"

# 演示默认用开集版（E04）：能同时演到 other 与拒识三态
DEMO_DEFAULT_CKPT = CKPT_DIR / "E04_broad_unfreeze3_best.pth"
DEMO_INTERVAL = 1.5             # 每张停留秒数（--interval 覆盖）

# ---------- 图像 / 归一化 ----------
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FILL = (124, 116, 104)          # 数据集中位色，旋转/填充用它而不是纯黑
MAX_ASPECT = 2.5                # 长宽比超过它就先 pad 成方形，避免裁掉主体

# ---------- 类别 ----------
# 主数据集（Kritik Seth）自带的水果目录名，恰好 10 类，不含草莓
FRUIT_CLASSES = [
    "apple", "banana", "grapes", "kiwi", "mango",
    "orange", "pear", "pineapple", "pomegranate", "watermelon",
]
OTHER_CLASS = "other"           # 开集负类（计划书 §4.4），训练时可加

# other 负类默认排除的目录名（蔬果难分的类别：番茄/黄瓜/甜椒/辣椒/茄子/牛油果/柠檬）。
# 它们如果被教成 other，模型的类别边界会变糊；统一留给 data/ood 做开集测试。
OTHER_EXCLUDE_DIRS = [
    "tomato", "cucumber", "bell pepper", "capsicum", "paprika",
    "chilli pepper", "jalepeno", "jalapeno", "eggplant", "avocado", "lemon",
]

CLASSES_CN = {
    "apple": "苹果",
    "banana": "香蕉",
    "grapes": "葡萄",
    "kiwi": "猕猴桃",
    "mango": "芒果",
    "orange": "橙子",
    "pear": "梨",
    "pineapple": "菠萝",
    "pomegranate": "石榴",
    "watermelon": "西瓜",
    "other": "其他 / 非目标",
}

# ---------- 划分 ----------
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # 对应计划书 v2 §3.4：合并后重划分

# ---------- 训练默认超参（= 计划书 §4.3 阶段一） ----------
DEFAULTS = dict(
    arch="mobilenet_v3_large",   # 或 mobilenet_v3_small
    head="linear",               # linear | mlp | official
    hidden=256,                  # 仅 mlp 头使用
    dropout=0.3,
    epochs=30,
    batch_size=32,
    lr_head=1e-3,
    lr_bb=1e-4,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=5,
    num_workers=4,
    unfreeze_last=0,             # 0 = 全程冻结；3 = 解冻 features[-3:]
    unfreeze_epochs=15,
)

# 置信/拒识阈值一律由 src/calibrate.py 标定后写进 checkpoint，
# 不要在这里手填（小数据微调模型的 softmax 过度自信，拍 0.6 往往永不触发）。
CONF_THRESHOLD = None


def ensure_dirs() -> None:
    """建好运行期需要的目录（幂等）。"""
    for d in (DATA_DIR, PROCESSED_DIR, CKPT_DIR, MODEL_DIR, LOG_DIR,
              EXP_DIR, DOCS_DIR, REAL_HOLDOUT_DIR, OOD_DIR,
              DEMO_SAMPLES_DIR, DEMO_OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def class_cn(name: str) -> str:
    """英文目录名 -> 中文名（找不到就原样返回）。"""
    return CLASSES_CN.get(name, name)
