# 受限算力下的可信代理模型优化 —— F1 赛车气动稳定性寻优（源代码）

《计算智能》课程设计 · 华南师范大学 人工智能学院

- 陈新安（202440012028）
- 郑骏远（202440012047）

---

## 一、项目简介

本项目研究**受限算力下的可信（Trustworthy）代理模型优化框架**：以 F1 赛车气动稳定性
寻优为案例，构建"代理建模 — 不确定性估计 — 风险敏感优化 — 数据精炼 — 景观诊断"流程，
并据此探讨"在代理模型驱动的优化中，数据覆盖度、不确定性管理与优化景观结构，
比优化器选择本身更决定最终优化质量"这一核心问题。

涉及的计算智能方法：粒子群优化（PSO，含自适应惯性权重与风险敏感变体）、
遗传算法（GA）、差分进化（DE）、模拟退火（SA）以及基于代理模型的优化。

> 数据说明：所用数据集为 Kaggle 公开数据集
> *F1 Aerodynamic Stability and Porpoising (150k Samples)*，
> 由一个 XGBoost 预测引擎生成（非真实 CFD 仿真）。
> 因此本项目关注的是\*\*代理优化的行为与可信性\*\*，而非对真实气动物理规律的拟合。

---

## 二、目录结构

```
source_code/
├── src/                      # 源代码（库）
│   ├── preprocessing/        # 数据清洗、归一化、特征工程
│   ├── models/               # M1–M6 代理模型（Ridge/RF/XGBoost/MLP/TabNet/DeepEnsemble）+ 评估
│   ├── optimization/         # PSO(base/adaptive/risk_sensitive) + GA + DE + SA + fitness + active_learning
│   ├── analysis/             # region_discovery / explainability(SHAP) / landscape_diagnosis
│   ├── scenarios/            # 多场景定义与执行
│   ├── visualization/        # EDA / 场景 / 区域 / Porpoising 绘图
│   └── utils/                # config.py（全局配置与路径）、metrics.py
├── experiments/              # 16 个 run_*.py 实验/驱动脚本（入口）
├── figures/                  # 报告最终结果图（32 张，便于无需运行即查看结果）
├── requirements.txt
└── README.md
```

> 说明：`src/` 与 `experiments/` 必须保持同级目录。`config.py` 以相对路径引用
> `data/`、`outputs/`、`figures/`、`reports/`；运行时缺失目录会被脚本自动创建。

---

## 三、环境配置

- Python 3.12（建议）
- 安装依赖：

```bash
pip install -r requirements.txt
# 若系统对全局安装有保护，可加 --break-system-packages
```

主要依赖：numpy、pandas、scikit-learn、scipy、matplotlib、seaborn、xgboost、torch、
shap、pytorch-tabnet。统一随机种子 `random_state=42`，结果可复现。

---

## 四、数据获取

最小源代码包不含数据集，请自行下载后放置到指定位置：

1. 下载 Kaggle 数据集
   *F1 Aerodynamic Stability and Porpoising (150k Samples)*：
   <https://www.kaggle.com/datasets/igormerlinicomposer/f1-aerodynamic-stability-and-porpoising-150k-samp/data>
2. 将原始 CSV 重命名/放置为：`data/raw/actaruslab_f1_telemetry_2026.csv`
3. 运行预处理脚本生成 `data/processed/`（见下）。

---

## 五、运行顺序（端到端）

所有脚本从包根目录运行，例如：`python experiments/run_preprocessing.py`

| 步骤 | 命令 | 主要产物 |
|----|------|------|
| 1 数据预处理 | `python experiments/run_preprocessing.py` | `data/processed/`（train/valid/test、scaler、样本权重） |
| 2 代理模型训练与对比 | `python experiments/run_model_comparison.py` | `outputs/models/`（M1–M6、DeepEnsemble）、`figures/03_*`、`04_*` |
| 3 PSO 变体与算法对比 | `python experiments/run_pso_comparison.py`、`python experiments/run_ci_comparison.py` | `figures/05_*`、`06_*`、`16_ci_comparison.png` |
| 4 多场景优化 | `python experiments/run_multi_scenario.py` | `outputs/scenarios/`、`figures/07_*`、pareto/平行坐标等 |
| 5 不确定性引导精炼 | `python experiments/run_active_learning.py` | 精炼后集成、`figures/active_learning_convergence.png` |
| 6 消融与区域发现 | `python experiments/run_ablation.py`、`python experiments/run_region_discovery.py` | `figures/14_*`、`15_*`、`robustness_heatmap.png` |
| 7 景观诊断 | `python experiments/run_landscape_diagnosis.py` | `figures/diagnosis_*`、诊断报告 |
| 8 Porpoising 风险 | `python experiments/run_porpoising_risk.py` | `figures/12_*`、`13_*`、`hessian_summary.png`、`risk_vs_optimal_overlay.png` |
| 9 OOD 压力测试（新） | `python experiments/run_ood_stress.py` | `figures/17_ood_stress.png`、`outputs/ood_stress_summary.csv` |
| 10 权重敏感性（新） | `python experiments/run_weight_sensitivity.py` | `figures/18_weight_sensitivity.png`、`outputs/weight_sensitivity_summary.csv` |
| 11 去饱和实验（新） | `python experiments/run_desaturation.py` | `figures/19_desaturation.png`、`outputs/desaturation_summary.csv` |

> 步骤 9–11 依赖步骤 1–2 产生的 `data/processed/` 与 `outputs/models/`（DeepEnsemble 及 scaler）。
> `experiments/run_visualization.py` 为统一可视化编排脚本，可在前述结果就绪后批量生成图表。

---

## 六、说明

- `figures/` 目录已附报告最终结果图，便于在不重新运行的情况下查看实验结论。
- 本包仅含源代码与结果图；数据集、训练好的模型与结题报告未随包提供，
  数据/模型可按上述步骤复现。
