# 项目文档索引

> 本文档为项目入口。详细内容请参阅 `docs/` 下的专项文档。

---

## 快速导航

| 你想... | 请阅读... |
|---------|----------|
| 了解项目背景和目标 | [`docs/doc-requirements.md`](docs/doc-requirements.md) |
| 了解总体技术路线和创新点 | [`docs/doc-plan.md`](docs/doc-plan.md) |
| 了解代码结构和技术栈 | [`docs/doc-architecture.md`](docs/doc-architecture.md) |
| 查看待执行任务 | [`docs/tasks/todo/`](docs/tasks/todo/) |
| 查看进行中任务 | [`docs/tasks/doing/`](docs/tasks/doing/) |
| 查看已完成任务 | [`docs/tasks/done/`](docs/tasks/done/) |
| AI Agent 快速上手 | [`AGENTS.md`](AGENTS.md) |

---

## 任务总览

| # | 任务 | 优先级 | 预计耗时 |
|---|------|--------|----------|
| T01 | [数据预处理与特征工程](docs/tasks/todo/T01_data_preprocessing.md) | 🔴 核心 | 2-3 天 |
| T02 | [探索性数据分析](docs/tasks/todo/T02_eda.md) | 🔴 核心 | 1-2 天 |
| T03 | [代理模型构建与对比](docs/tasks/todo/T03_surrogate_model.md) | 🔴 核心 | 3-5 天 |
| T04 | [PSO 寻优框架搭建](docs/tasks/todo/T04_pso_optimization.md) | 🔴 核心 | 3-4 天 |
| T05 | [主动学习迭代精炼](docs/tasks/todo/T05_active_learning.md) | 🟢 可选 | 2-3 天 |
| T06 | [多场景优化与可解释性](docs/tasks/todo/T06_multi_scenario.md) | 🔴 核心 | 2-3 天 |
| T07 | [Porpoising 风险可解释性](docs/tasks/todo/T07_porpoising_risk.md) | 🟡 推荐 | 1-2 天 |
| T08 | [实验设计与评价](docs/tasks/todo/T08_experiments.md) | 🔴 核心 | 2-3 天 |
| T09 | [结果可视化汇总](docs/tasks/todo/T09_visualization.md) | 🔴 核心 | 1-2 天 |
| T10 | [结题报告撰写](docs/tasks/todo/T10_final_report.md) | 🔴 核心 | 2-3 天 |

---

## 创新点

1. **风险敏感 PSO** — DeepEnsemble 提供不确定性，适应度 μ(x)-λσ(x) 同步解决 OOD/不平衡/幻觉
2. **主动学习闭环** — PSO 探索反馈不确定性区域，增量精炼代理模型
3. **Porpoising 风险热力图** — 代理模型梯度识别海豚跳高风险区
4. **多场景可解释性体系** — SHAP + 雷达图 + 约束灵敏度 + 反事实分析
5. **6 模型系统对比** — 从线性回归到 DeepEnsemble 的完整谱系

---

## 历史草案

| 文件 | 说明 |
|------|------|
| `docs/archive/task0.md` | 第 0 版任务草案 (偏工程) |
| `docs/archive/task1.md` | 第 1 版任务草案 (偏学术) |
| `docs/archive/opencode-usage-guide.md` | OpenCode 使用指南参考 |
