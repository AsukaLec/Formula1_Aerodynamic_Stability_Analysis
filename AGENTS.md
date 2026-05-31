# AGENTS.md

> 本文档供 AI Agent (OpenCode/Claude Code 等) 和开发者共同使用。
> 每次新 Session 开始时，Agent 应首先读取此文件及 `docs/` 下的核心文档。

---

## 项目概述

- **课题**: 《计算智能》课程设计 — 基于粒子群算法的 F1 赛车气动参数寻优方法研究
- **目标**: 建立代理模型 + PSO 的轻量化气动优化框架，实现不同工况下最优参数快速求解
- **语言**: Python 3
- **环境**: 常规 PC，无需 GPU (但建议有)

---

## 文档驱动开发

本项目采用 **文档驱动开发** 模式。代码是副产品，文档才是真成果。

### 核心文档 (必读)

| 文档 | 路径 | 说明 |
|------|------|------|
| 需求文档 | `docs/doc-requirements.md` | 项目背景、数据、核心需求、量化目标 |
| 架构文档 | `docs/doc-architecture.md` | 代码结构、数据流、技术栈、模型架构 |
| 方案文档 | `docs/doc-plan.md` | 总体路线、创新点、任务依赖、风险预案 |

### 任务管理

```
docs/tasks/
├── todo/          # 待执行任务
├── doing/         # 正在执行 (同时只有一个)
└── done/          # 已完成
```

- 执行任务时: 从 `todo/` 移动到 `doing/`
- 任务完成后: 从 `doing/` 移动到 `done/`，更新任务文档中的关键决策记录
- 每个任务文档包含: 描述、验收标准、输入/输出、弹性空间

---

## 技术栈

| 类别 | 工具 |
|------|------|
| 数据处理 | pandas, numpy |
| 可视化 | matplotlib, seaborn |
| 传统 ML | scikit-learn, xgboost |
| 深度学习 | PyTorch |
| 优化算法 | 自定义 PSO |
| 可解释性 | SHAP |

---

## Python 环境

- 使用主机 Windows Python: `/mnt/e/python313/python.exe`
- 所有依赖已预装 (numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, xgboost, torch, shap, joblib)，无需新建虚拟环境
- 运行脚本格式: `/mnt/e/python313/python.exe <script.py>`

## 当前状态

- 项目处于 **T06 已完成**, T07/T08 待开始
- `src/preprocessing/` 已完成 (cleaner, normalizer, feature_engineering)
- `src/models/` 已完成 (baseline, mlp, tabnet_model, deep_ensemble, evaluate)
- `src/optimization/` 已完成 (pso_base, pso_adaptive, pso_risk_sensitive, fitness)
- `src/analysis/` 已完成 (region_discovery, explainability)
- `src/scenarios/` 已完成 (scenario_def, scenario_runner)
- `src/visualization/` 已完成 (plot_eda, plot_regions, plot_scenarios)
- `src/utils/config.py` 已创建
- `data/raw/` 已有原始数据集
- T01, T02, T03, T04, T05, T06 任务文档已移至 `docs/tasks/done/`

---

## 开发约定

1. **先读文档** — 每次新 Session 先读 `docs/doc-plan.md` 了解全局
2. **一个任务 = 一个 Session** — 不要在一个 Session 中跨越多个任务
3. **文档实时更新** — 任务执行中发现的关键决策必须记录在任务文档中
4. **代码放在 src/** — 按照架构文档的模块结构组织
5. **实验脚本放在 experiments/** — 不污染 src/
6. **随机种子统一** — `random_state=42`，确保可复现
7. **弹性优先** — 每个任务有核心/推荐/可选三级，时间不足时优先保障核心项
8. **文档同步更新** — 修改全局常量（阈值、标签、配置键名等）后，必须搜索 `docs/` 下所有引用该值的文档并一并更新，不得只改代码而不同步文档
9. **英文输出** — 开发阶段所有代码 print 语句、中间报告文字（如 `reports/*.md`）一律使用英文。最终结题报告 (T10) 再切换为中文，避免 Windows Python 与 WSL/Linux 间编码不一致导致乱码。但是在对话时请使用中文。
