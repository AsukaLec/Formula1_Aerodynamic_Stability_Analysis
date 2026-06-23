# T17 — Story-Driven Full Report Rewrite (report_v2)

## 描述

将报告从「多个独立实验的拼盘」重写为围绕 **可信代理模型优化 (Trustworthy Surrogate-Driven Optimization)** 的统一调查主线，输出到全新独立文件夹 `report_v2/`（不动旧 `latex_report/`、`reports/`、PPT）。

依据：`redo_target1.md`（故事重构规范）、`further.md`（蓝图终审 6 项修订）、`story_answers.md` / `story_answers_r2.md`（两轮理解校验）、`story_refactor_plan.md`（执行蓝图）。

## Core Claim（全文唯一主张，所有实验须服务于此）

> 在代理模型驱动优化中，数据覆盖度、不确定性管理、优化景观结构比优化器选择本身更决定最终优化质量。

## 七幕 / 章节问题表

| tex | 回答的唯一问题 |
|-----|------|
| 02_introduction | 为何主线是「可信优化」 |
| 03_dataset | 数据隐患在哪 |
| 04_surrogate | 代理够准了吗（压缩 bake-off） |
| 05_anomalies | 为什么开始怀疑代理模型 |
| 06_uncertainty | 代理模型是否可信 |
| 07_risk_sensitive | 如何避免优化器利用盲区（含 OOD 压力测试） |
| 08_refinement | 这些盲区是否真实存在 |
| 09_similar_solutions | 为何不同优化器/场景得到相似解（含权重敏感性） |
| 10_landscape | 异常是否有统一根因（含去饱和实验，高潮） |
| 11_conclusion | 工程启示（Contributions vs Findings + Porpoising 案例） |

## 关键设计决策（不可推翻）

- Landscape Diagnosis = **Key Finding**，非 Contribution；非全文唯一支柱。
- Contributions = 团队「做了什么」（含可信优化诊断流程）；Findings = 团队「发现了什么」。
- Risk-Sensitive PSO = 防御机制，非核心创新。
- Porpoising = Case Study（框架应用价值），不推进主线，置于结论/讨论章。
- 不动点主线：「何时能相信最优解」比「找到最优解」更根本（不绑定任何单一根因）。
- 沿用 6 轮锁定术语：加权标量化≠NSGA-II、单轮→（本轮允许）有限多轮、可靠性约束寻优、SHAP=统计关联非因果（共线 r=0.895）。

## 新实验（已完成，结果入报告）

| 脚本 | 图 | 核心结果 |
|------|----|------|
| experiments/run_ood_stress.py | 17_ood_stress.png | vanilla μ 幻觉至 101.15（基盆边 27.6%），μ−λσ 将幻觉率压至 0%；σ 在边界区峰值 13.95 |
| experiments/run_weight_sensitivity.py | 18_weight_sensitivity.png | 36 个权重向量下 wing/drag 位移 0%，downforce 仅 0.6%（权重失效） |
| experiments/run_desaturation.py | 19_desaturation.png | 单调再标度下稳定性固定（极差 0.87）但最优位置游移（downforce 2849N, wing 10.7°）；V4 注入梯度使 wing 35°→20°（优化器有能力） |

## 验收标准

- 七幕主线可复述；每章只答一问。
- PSO/GA/DE、多场景同解被叙述为异常证据而非性能优胜。
- Landscape 为高潮 Finding；Contributions 与 Findings 严格分离。
- 三张新图入册；`report_v2/` 自包含、XeLaTeX 可编译。
- 6 轮锁定决策无违反。

## 状态

- [x] P0 骨架 + 任务文档
- [x] P1 三个新实验（OOD / 权重敏感性 / 去饱和）
- [x] P2 章节重写（00–11 共 12 个 tex，全部叙事重写）
- [x] P3 figures 整理（32 张 PNG，含新增 17/18/19，剔除 zip/md 污染）
- [x] P4 静态校验（环境/括号平衡、\input 与文件一一对应、15 张被引图均就位）

## 编译说明

本机无 TeX 工具链，未做本地 XeLaTeX 编译。`report_v2/` 复用 `latex_report/`
（Overleaf 可编译）的同款 preamble（ctexart + XeLaTeX），仅新增标准包 `mdframed`
（用于章节问题框）。已通过静态校验，**可原封不动上传 Overleaf 编译**。
