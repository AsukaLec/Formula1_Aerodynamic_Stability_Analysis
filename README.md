# Formula1 Aerodynamic Stability Analysis

基于代理模型与改进粒子群算法的 F1 赛车气动参数寻优方法研究

## 快速开始

```bash
pip install -r requirements.txt
```

## 文档

| 文档 | 说明 |
|------|------|
| [`AGENTS.md`](AGENTS.md) | AI Agent 快速上手 |
| [`docs/doc-requirements.md`](docs/doc-requirements.md) | 需求文档 |
| [`docs/doc-architecture.md`](docs/doc-architecture.md) | 架构与代码结构 |
| [`docs/doc-plan.md`](docs/doc-plan.md) | 总体方案与创新点 |
| [`docs/tasks/todo/`](docs/tasks/todo/) | 任务列表 |

## 项目结构

```
Formula1_Aerodynamic_Stability_Analysis/
├── data/                    # 数据集
├── docs/                    # 项目文档
│   └── tasks/               # 任务管理 (todo/doing/done)
├── src/                     # 源代码
│   ├── preprocessing/
│   ├── models/
│   ├── optimization/
│   ├── scenarios/
│   └── visualization/
├── experiments/             # 实验脚本
├── notebooks/               # Jupyter
├── outputs/                 # 模型与结果
├── figures/                 # 图表
└── reports/                 # 实验报告
```

## 数据集

[Kaggle - F1 Aerodynamic Stability and Porpoising](https://www.kaggle.com/datasets/igormerlinicomposer/f1-aerodynamic-stability-and-porpoising-150k-samp/data)
