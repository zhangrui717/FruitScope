# FruitScope 水果观察站

FruitScope 面向中学生的课堂和日常观察。上传一张水果照片，或点「试试示例」选一张照片，再点「开始识别」。结果页展示模型最倾向的类别、得分、前三个候选和水果观察笔记。
网页调用 `src.predict.Predictor`，沿用训练代码的类别顺序、预处理、温度校准和拒识阈值。

## 本地运行网站

使用标准 CPython 3.12，在仓库根目录执行：

```bash
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py
```

默认使用随仓库提供的 `model/fruitscope.pth`，由校准后的 E04 权重导出，仅保留网络权重与推理参数。自训权重可放在 `ckpt/E04_broad_unfreeze3_best.pth`，或通过 `FRUITSCOPE_CHECKPOINT` 指定。

接入后先用 `demo/samples/01_correct_apple.jpg` 对照终端输出和网页结果：

```bash
python -m src.predict --ckpt model/fruitscope.pth --image demo/samples/01_correct_apple.jpg
```

如果权重沿用原来的 `ckpt/E04_broad_unfreeze3_best.pth` 路径，把命令中的 `--ckpt` 参数改为该路径。

## 部署网站

在 [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) 连接 GitHub，选择 `web/app.py` 作为入口，选择 Python 3.12。网站依赖由同目录的 `web/requirements.txt` 提供，项目根目录的依赖清单用于训练。

部署会使用仓库里的 E04 推理权重。部署完成后，用未登录的浏览器分别验证一张示例图和一张自行上传的照片。平台休眠后，可点击唤醒再体验。

## 数据与结果

训练与示例图来自 Kritik Seth 在 Kaggle 发布的 [Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)（数据集页面标注 CC0）。完整数据请从数据集页面下载。项目只包含少量演示图，来源和预期标签见 [演示图清单](demo/manifest.json)。

训练记录中，E04 在 180 张测试图上的准确率为 **90.56%**，宏平均 F1 为 **0.8990**。它能对部分陌生图片给出「其他」或低分提示。一个真实失败例子是把范围外的柠檬判为橙子，模型得分约 99.9%。两者颜色和外形相近可能是原因。详见 [实验记录](docs/experiments.md) 与 [模型训练报告](docs/模型训练报告.md)。

AI 辅助用于网站实现、模型接入、交互测试和发布准备。训练代码、E04 权重和实验记录来自项目已有的训练产物。

## 目录

```
fruitscope/
├── config.py                  # 路径、类别、超参、增强常量的唯一来源
├── web/                       # Streamlit 网页、样式、科普内容和部署依赖
├── src/
│   ├── transforms.py          # 训练增强 / 评估预处理 / 上线读图三件套
│   ├── dataset.py             # 只从 data/splits.json 读数据
│   ├── model.py               # MobileNetV3 + linear/mlp/official 三档头 + BN 锁定
│   ├── train.py               # 两阶段训练、按 macro-F1 早停、自动写 runs.csv
│   ├── eval.py                # acc+95%CI / macro-F1 / Top-3 / ECE / 混淆矩阵
│   ├── calibrate.py           # 温度标定 + 拒识阈值标定（回写 checkpoint）
│   ├── ood.py                 # other 负类采样、MSP/Energy、拒识率曲线
│   ├── predict.py             # 推理参考实现（top-k + 拒识 + --selfcheck）
│   ├── dedup.py               # 纯 numpy pHash 去重与跨划分泄漏检查
│   └── utils.py               # set_seed / Wilson CI / runs.csv / 日志
├── scripts/01_data_prepare.py # 合并 + 去重 + 70/15/15 重划分 + 清洗报告
├── tests/smoke_test.py        # 合成小数据集端到端冒烟测试（无数据集也能跑）
├── data/                      # raw/ 原始、processed/ 整理后、splits.json、real_holdout/
├── ckpt/ model/ logs/ experiments/runs.csv docs/
```

## 快速开始（服务器）

```bash
# 在仓库根目录运行，先激活自己的 Python / PyTorch 训练环境

# 0) 环境自检（写 docs/env.md 与 docs/env_lock.txt）
python scripts/00_env_check.py

# 1) 依赖（若换环境）
pip install -r requirements.txt

# 2) 数据：解压主数据集（Kaggle Fruit and Vegetable Image Recognition）
#    结构是 train|validation|test/<class>/，脚本会合并后按 70/15/15 重新划分
unzip archive.zip -d data/raw/fruit-veg
python scripts/01_data_prepare.py --src data/raw/fruit-veg        # 10 类，~840 张

# 3) 训练（阶段一：冻结 backbone 只训头）
python -m src.train --tag E01_mbv3l_linear --epochs 30

# 4) OOD 集 + 校准 + 阈值（番茄/黄瓜/甜椒/辣椒/茄子/柠檬，模型没见过）
python scripts/02_build_ood.py --src data/raw/fruit-veg --per-class 25
python -m src.calibrate --ckpt ckpt/E01_mbv3l_linear_best.pth --update-ckpt

# 5) 评估与坏案例
python -m src.eval --ckpt ckpt/E01_mbv3l_linear_best.pth --which test \
    --confusion-png docs/curves/cm_E01.png --save-json docs/E01_test.json
```

## 一键跑完（服务器推荐）

```bash
# 在仓库根目录运行
SRC=data/raw/fruit-veg bash scripts/03_run_experiments.sh      # 10类 + 11类 + 校准 + 指标
SRC=data/raw/fruit-veg EPOCHS=30 OTHER_TOTAL=450 bash scripts/03_run_experiments.sh   # 覆盖参数
```

## other 负类（11 类开集）怎么加

蔬菜做负类方向对（同属食物，比汽车家具更能教边界），但四条必须同时满足：

| 复核项 | 落地方式 |
| --- | --- |
| 排除蔬果难分 | 默认排除 `tomato/cucumber/bell pepper/capsicum/paprika/chilli pepper/jalepeno/eggplant/avocado/lemon`（`config.OTHER_EXCLUDE_DIRS`，可用 `--other-exclude` 改）；这些留给 `data/ood` |
| 总量 300–500 | `--other-total 450`，按目录数均分额度（26 个可用目录 → 每类 17 张 = 442） |
| 防止 other 主导 | 11 类那次训练必须 `--weighted-sampler`；`01_data_prepare.py` 会打印"最大/最小类比"和 other 占比，>40% 直接告警 |
| 同步划分 + 防泄漏 | other 与正类一样按 70/15/15 切分（同一 `split_class`），并参与 pHash 去重与跨划分泄漏检查 |

**实测教训（重要）**：只把 17 个根茎类蔬菜当负类（窄口径），OOD 集里**只有 29%** 被正确判成 other，
其余 71% 被硬判成 apple/orange —— 番茄、柠檬这种"长得像水果"的负类根本没教过，边界学不到。
改用宽口径（蔬果难分也进 other + OOD 只用不重叠图片）后升到 **72%**，再加解冻 features[-3:] 到 **81.3%**。
详见 `docs/experiments.md`。

```bash
# 宽口径：蔬果难分也进 other；OOD 用"未被 other 占用"的同类别图片（manifest 排除，保证不重叠）
python scripts/01_data_prepare.py --src data/raw/fruit-veg \
    --out data/processed_other_broad --split data/splits_other_broad.json \
    --report docs/cleaning_report_other_broad.md \
    --other-from data/raw/fruit-veg --other-total 450 --other-include-ambiguous
python scripts/02_build_ood.py --src data/raw/fruit-veg --per-class 25 --clean \
    --exclude-from data/manifest_other_broad.csv

# 训练（解冻 features[-3:] 版本实测最好）
python -m src.train --tag E04_broad_unfreeze3 --split data/splits_other_broad.json \
    --epochs 30 --unfreeze-last 3 --unfreeze-epochs 15 --weighted-sampler

# 训练完照常校准 + 标阈值；会打印 OOD→other 比例并回填 runs.csv
python -m src.calibrate --ckpt ckpt/E04_broad_unfreeze3_best.pth \
    --split data/splits_other_broad.json --update-ckpt
```

一键跑：`bash scripts/04_broad_other_pipeline.sh`

`src/predict.py` 的三种文案（`_message`）：拒识 →"未识别为支持的水果…"；
命中 `other` →"这不是我认识的水果（支持苹果/香蕉/…/西瓜）"；其余 →"可能是苹果"。

## 终端演示（一条命令看效果，不用网页/不用上传）

素材是**固定副本**：`demo/samples/` + `demo/manifest.json`，只从测试集 / OOD 集挑（**不含训练图**），
由脚本按当前 checkpoint 现场判定后写入，别手工往里丢文件：

| kind | 张数 | 用途 |
| --- | --- | --- |
| `correct` | 10 | 10 个水果类各 1 张"判对"，覆盖全部支持类别 |
| `other` | 2 | 甜椒 / 菠菜这类非支持类，应出"这不是我认识的水果" |
| `reject` | 1 | 测试集里置信度最低的一张（0.355 < 阈值 0.479）|
| `misclass` | 1 | 已知局限：柠檬被判成橙子（0.999），诚实展示 |

```bash
python scripts/demo.py                       # 交互播放：每张停 1.5s + ANSI 颜色
python scripts/demo.py --interval 3           # 放慢
python scripts/demo.py --no-wait --no-color   # 一口气跑完、纯文本
python scripts/demo.py --ckpt ckpt/E01_mbv3l_linear_best.pth   # 换 10 类模型
python scripts/demo.py --no-save              # 只打终端，不生成标注图
```

非 TTY（`| tee`、`> log`、`nohup`）时**自动**不等待、不输出 ANSI 颜色，可直接丢日志：

```bash
python scripts/demo.py > demo/output/demo.log 2>&1     # 实测日志里 0 个 ESC 字节
```

实际输出（服务器 E04，`--no-wait`，节选）：

```text
FruitScope 演示  11 类 · 14 张素材
  checkpoint  ckpt/E04_broad_unfreeze3_best.pth
  校准温度    T=0.512    拒识阈值 0.479    设备 cuda
  运行模式    静默模式（不等待/无颜色）

────────────────────────────────────────────────────────────────────────────
[01/14] 01_correct_apple.jpg   预期 苹果(apple) · 正确命中
  预测   苹果(apple)  1.000   ✓ 预期判对
  Top3   苹果(apple)           1.000  ██████████████████████████
         其他 / 非目标(other)  0.000  ··························
         石榴(pomegranate)     0.000  ··························
  三态   识别水果
  文案   可能是苹果
  说明   测试集 apple 图，判对，置信度 1.000
  标注图   demo/output/01_correct_apple.png
...
[11/14] 11_other_other.jpeg   预期 其他 / 非目标(other) · 非目标(other)
  预测   其他 / 非目标(other)  1.000   ✓ 预期判成 other
  三态   非目标(other)
  文案   这不是我认识的水果（支持苹果/香蕉/葡萄/猕猴桃/芒果/橙子/梨/菠萝/石榴/西瓜）
...
[13/14] 13_reject_mango.jpg   预期 芒果(mango) · 低置信拒识
  预测   梨(pear)  0.355   ✓ 预期被拒识
  Top3   梨(pear)              0.355  █████████·················
         芒果(mango)           0.326  ████████··················
         苹果(apple)           0.210  █████·····················
  三态   拒识（阈值 0.479）
  文案   未识别为支持的水果，请换一张更清晰、主体更完整的照片
...
[14/14] 14_misclass_lemon.jpg   预期 lemon(lemon) · 边界误判
  预测   橙子(orange)  0.999   ✓ 预期仍判错（诚实展示局限）
  三态   识别水果
  文案   可能是橙子
  说明   OOD 集原图是 lemon（模型未训练过该类别），被判成 orange，置信度 0.999

────────────────────────────────────────────────────────────────────────────
汇总  14 张素材 · checkpoint E04_broad_unfreeze3_best.pth
  预期类型  正确命中 10   非目标(other) 2   低置信拒识 1   边界误判 1
  三态次数  识别水果 11   非目标(other) 2   拒识 1
  复现预期  14/14（100.0%）
  水果类准确率  10/11 = 90.9%   （只算 manifest 里预期为水果的那些）
  全部素材都复现了预期 ✓
  总耗时      6.2s（439 ms/张）
  标注图/汇总 demo/output/  （拉回本机: scp -r <host>:demo/output .）
```

标注图写到 `demo/output/`（`matplotlib` 强制 Agg，绝不弹窗），`summary.json` 同步留档，方便截图/下载。

换模型后重建素材，保证演示与当前 checkpoint 一致：

```bash
python scripts/05_build_demo.py                              # 默认 E04 + splits_other_broad
python scripts/05_build_demo.py --ckpt ckpt/E01_mbv3l_linear_best.pth \
    --split data/splits.json                                  # 10 类版（没有 other 案例）
python scripts/05_build_demo.py --dry-run                     # 只看会挑哪些图，不落盘
```

约定：

- 只用 test/OOD 选图，绝不碰 train；先推理再挑，保证"10 类各 1 张判对"；
- 同一 checkpoint + 同一划分 ⇒ 同一批图（挑选过程无随机）；
- 推理、校准温度、拒识阈值、三态文案全部来自 `src/predict.py::Predictor` 与 checkpoint，
  **没有第二套推理逻辑**，所以演示数字与 `docs/experiments.md` 一致；
- 不新增依赖（torch / torchvision / Pillow / numpy / matplotlib 已在 requirements）。

## 实测结果（2026-11-04，4×RTX 3090）

完整记录见 [`docs/experiments.md`](docs/experiments.md)；本轮实测指标以 `ckpt/*_calib.json` 与 `docs/E0*_test.json` 为准
（`experiments/runs.csv` 表头已就位，历史几次 run 的追加写入在首轮实验之后才接上）。
**模型结构、训练过程与踩坑记录**（含坏案例分析）见 [`docs/模型训练报告.md`](docs/模型训练报告.md)。

| run | 数据 | 冻结策略 | test acc | macro-F1 | 水果类 acc | OOD→other |
| --- | --- | --- | --- | --- | --- | --- |
| E01 | 10 类 | 全程冻结 | 0.9576 | 0.9572 | **0.9576** | 无 other 类 |
| E02 | 11 类窄口径 | 全程冻结 | 0.8920 | 0.8872 | 0.9237 | 28.9% |
| E03 | 11 类宽口径 | 全程冻结 | 0.8667 | 0.8680 | 0.9068 | 72.0% |
| **E04** | **11 类宽口径** | **解冻 [-3:] 15 轮** | **0.9056** | **0.8990** | 0.9153 | **81.3%** |
| E05 | 10 类 | 解冻 [-3:] 15 轮 | 0.9576 | 0.9572 | 0.9576 | 无 other 类 |

- 数据：1033 张原始 → pHash 去重 787 张（丢掉 246 张跨 train/test 的重复图）→ 551/118/118；
- 温度校准把 ECE 从 0.20–0.22 压到 0.04–0.08（T≈0.43–0.52），阈值写进 checkpoint（E01 标出来是 0.51，不是 0.6+）；
- 10 类模型解冻没有收益（E05 == E01，两次都在同一轮早停）；11 类解冻收益明显（E03→E04）；
- 混淆矩阵需要 `matplotlib`（已在 requirements），否则降级存 `.txt`。

## 关键约定（别改）

1. **划分只认 `data/splits.json`**：所有实验读同一个文件，指标才可比。
2. **test 只看一次**：模型选择与早停都用 val 的 macro-F1。
3. **BN 锁定**：`model.train()` 之后必须再调 `set_frozen_bn_eval()`，否则冻结等于没冻结。
4. **阈值不手填**：一律由 `src/calibrate.py` 标定后写进 checkpoint。
5. **预处理唯一**：训练/评估/推理都用 `src/transforms.py`。

## 本地（Windows，无 GPU）跑冒烟测试

```powershell
python tests/smoke_test.py
```

脚本会造一个 3 类、每类 24 张的合成数据集，走完 准备 -> 训练 2 epoch -> 评估 -> 校准 -> 推理自检，
用于确认代码链路没断（真实精度无意义）。

## 目录清单

| 文件 | 作用 | 对应计划书 |
| --- | --- | --- |
| `scripts/00_env_check.py` | 环境自检 → `docs/env.md` / `env_lock.txt` | §4.2 D1 |
| `scripts/01_data_prepare.py` | 合并 + pHash 去重 + 70/15/15 重划分 + other 负类（配额/排除/失衡告警） | §3.4 §3.5 §4.4 |
| `scripts/02_build_ood.py` | 用"蔬果难分"类构建 `data/ood` 开集测试集 | §3.5 §4.4 |
| `scripts/03_run_experiments.sh` | 服务器一键：10类/11类数据 → 训练 → 校准 → 指标 | §4.3 §4.4 |
| `scripts/04_broad_other_pipeline.sh` | 宽口径 other（含蔬果难分）+ 不重叠 OOD + 重标定 | §4.4 |
| `scripts/05_build_demo.py` | 从 test/OOD 挑固定演示素材（10类各1张判对 + other + 拒识 + 误判）→ `demo/` | §5 |
| `scripts/demo.py` | 终端逐张演示：n/N 进度、Top3 条形、三态文案、Agg 标注图 + 汇总 | §5 |
| `src/train.py` | 两阶段训练、macro-F1 早停、runs.csv | §4.3 §4.7 |
| `src/eval.py` | acc+CI / macro-F1 / Top-3 / ECE / 混淆矩阵 | §4.4 |
| `src/calibrate.py` | 温度 + 拒识阈值标定 | §4.4 |
| `src/ood.py` | other 负类、MSP/Energy、拒识曲线 | §4.4 §4.6 |
| `src/predict.py` | top-k + 拒识 + `--selfcheck` | §4.5 |

> 注：docs/env.md 与 docs/env_lock.txt 是**当前机器**的快照，换机器（本地 CPU <-> 服务器 4x3090）后请重跑 scripts/00_env_check.py 覆盖。
