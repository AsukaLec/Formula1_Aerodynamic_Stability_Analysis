# 架构文档

## 代码结构

```text
Formula1_Aerodynamic_Stability_Analysis/
│
├── data/                              # 数据目录
│   ├── raw/                           # 原始数据集
│   │   └── actaruslab_f1_telemetry_2026.csv
│   └── processed/                     # 预处理后数据
│       ├── train.csv, valid.csv, test.csv
│       └── scaler.pkl
│
├── docs/                              # 项目文档
│   ├── doc-requirements.md            # 需求文档
│   ├── doc-architecture.md            # 本文件
│   ├── doc-plan.md                    # 方案文档
│   ├── tasks/                         # 任务管理
│   │   ├── todo/                      # 待办任务
│   │   ├── doing/                     # 进行中
│   │   └── done/                      # 已完成
│   └── archive/                       # 历史文档归档
│
├── notebooks/                         # Jupyter 探索性笔记本
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_prototype.ipynb
│
├── src/                               # 源代码
│   ├── preprocessing/                 # 数据预处理
│   │   ├── cleaner.py
│   │   ├── normalizer.py
│   │   └── feature_engineering.py
│   ├── models/                        # 代理模型
│   │   ├── baseline.py                # M1-M3 传统模型
│   │   ├── mlp.py                     # M4 BP 神经网络
│   │   ├── tabnet_model.py            # M5 TabNet (可选)
│   │   ├── deep_ensemble.py           # M6 DeepEnsemble
│   │   └── evaluate.py                # 统一评估框架
│   ├── optimization/                  # PSO 优化
│   │   ├── pso_base.py                # 标准 PSO
│   │   ├── pso_adaptive.py            # 自适应 PSO
│   │   ├── pso_risk_sensitive.py      # 风险敏感 PSO
│   │   ├── fitness.py                 # 适应度函数
│   │   └── active_learning.py         # 主动学习 (可选)
│   ├── scenarios/                     # 多场景
│   │   ├── scenario_def.py
│   │   └── scenario_runner.py
│   ├── visualization/                 # 可视化
│   │   ├── plot_eda.py
│   │   ├── plot_results.py
│   │   └── plot_porpoising.py
│   └── utils/                         # 工具
│       ├── config.py
│       └── metrics.py
│
├── experiments/                       # 实验脚本
│   ├── run_model_comparison.py
│   ├── run_pso_comparison.py
│   └── run_ablation.py
│
├── outputs/                           # 运行输出
│   ├── models/                        # 训练好的模型
│   └── scenarios/                     # 各场景寻优结果
│
├── figures/                           # 图表
│   ├── eda/
│   ├── models/
│   ├── pso/
│   ├── scenarios/
│   └── porpoising_risk/
│
├── reports/                           # 报告
│   ├── statistics_report.md
│   ├── model_comparison.csv
│   ├── imbalance_report.md
│   └── experiment_results.md
│
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## 技术栈

| 类别 | 主选 | 备选 |
|------|------|------|
| 数据处理 | pandas, numpy | polars |
| 可视化 | matplotlib, seaborn | plotly |
| 传统 ML | scikit-learn | — |
| Boosting | xgboost | lightgbm, catboost |
| 深度学习 | PyTorch | TensorFlow/Keras |
| PSO | 自定义实现 | pyswarms (原型) |
| 可解释性 | SHAP | LIME, Permutation Imp. |
| 实验管理 | 手动脚本 | MLflow |
| 代码托管 | GitHub | — |

---

## 数据流

```text
raw/actaruslab_f1_telemetry_2026.csv
    │
    ├──→ cleaner.py ──→ processed/clean_dataset.csv
    │                          │
    ├──→ normalizer.py ────────┤
    │                          │
    ├──→ feature_engineering ──┤
    │                          ↓
    │              processed/{train,valid,test}.csv
    │                          │
    │              ┌───────────┴───────────┐
    │              ↓                       ↓
    │      baseline.py               mlp.py
    │      (M1-M3)                   deep_ensemble.py
    │              │                       │
    │              └───────────┬───────────┘
    │                          ↓
    │              evaluate.py → model_comparison.csv
    │                          ↓
    │              outputs/models/best_model.pt
    │                          │
    │              ┌───────────┴───────────┐
    │              ↓                       ↓
    │      pso_*.py                 scenario_runner.py
    │      fitness.py               ─── outputs/scenarios/
    │              │
    │              ↓
    │      experiments/ → experiment_results.md
    │              │
    │              ↓
    │      plot_*.py → figures/
    │              │
    │              ↓
    │      结题报告
```

---

## 模型架构参考 (MLP)

```text
Input(5) → Linear(128) → ReLU → Dropout(0.2)
         → Linear(256) → ReLU → Dropout(0.2)
         → Linear(128) → ReLU → Dropout(0.2)
         → Linear(64)  → ReLU
         → Linear(1)   → Output
```

- 优化器: Adam (lr=1e-3, weight_decay=1e-5)
- 损失: 加权 MSE (低分样本权重加倍)
- Early Stopping: patience=20, monitor=val_loss
- 超参数可调: 层数、隐藏单元数、dropout 率、lr

## DeepEnsemble

M 个独立 MLP (M=3~5)，仅初始化 seed 和数据 shuffle 不同。输出 μ(x) 和 σ(x)。
