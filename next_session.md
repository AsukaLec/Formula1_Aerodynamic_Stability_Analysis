# Next Session — 会话交接文档

> 本文档用于指导下一个 AI Agent Session 快速理解项目状态。
> 请在新 Session 开始时**首先读取此文件**及 `AGENTS.md`。

---

## 一、项目概述

- **课题**: 《计算智能》课程设计 — 基于代理模型与改进粒子群算法的 F1 赛车气动稳定性优化研究
- **团队**: 陈新安 (202440012028)、郑骏远 (202440012047)
- **学校**: 华南师范大学 人工智能学院
- **仓库**: https://github.com/AsukaLec/Formula1_Aerodynamic_Stability_Analysis
- **语言**: Python 3
- **当前阶段**: 结题报告历经 6 轮学术审查修正，已定稿；PPT 已完成

---

## 二、必读文档 (按优先级排序)

| 文档 | 路径 | 说明 |
|------|------|------|
| AGENTS.md | `AGENTS.md` | 开发约定、Python 环境、当前状态 |
| 方案文档 | `docs/doc-plan.md` | 总路线、创新点、进度表 |
| 需求文档 | `docs/doc-requirements.md` | 问题定义、数据、量化目标 |
| 架构文档 | `docs/doc-architecture.md` | 代码结构、技术栈 |
| 结题报告 | `latex_report/main.tex` | 主文件，`\input` 11 个章节 |
| 任务目录 | `docs/tasks/done/` | T01 - T16 所有任务文档 |

---

## 三、当前任务完成状态

| 任务 | 状态 | 说明 |
|:---:|:---:|------|
| T01 - T10 | ✅ | 初始开发完成 (数据预处理 → 建模 → PSO → 多场景 → 可视化 → 结题报告) |
| T11 | ✅ | 第一轮修正: 补 CI 对比 (GA/DE/SA)、K-Fold 验证、路线调整说明、术语修正 |
| T12 | ✅ | 第二轮修正: 场景定义表硬错误修复 (S1 wing [0,15] → [20,29])、措辞收敛 |
| T13 | ✅ | 第三轮修正: 学术表达精确化 (核心创新→核心改进、OOD 定义注、加权标量化) |
| T14 | ✅ | 第四轮修正: 方法论证据链 (Pareto 术语全改、因果限定、DeepEnsemble/XGBoost 角色说明) |
| T15 | ✅ | 第五轮修正: 结论层次提升 (Landscape 前移至贡献首位、RF=1.0 结构规律性解释) |
| T16 | ✅ | 第六轮修正: 因果论证整合 (第9章→全篇统一解释框架、闭环→单轮反馈、DRS 失效诊断) |
| PPT | ✅ | 23 页答辩 PPT 已生成, 发言稿已输出 |

---

## 四、核心目录结构

```
Formula1_Aerodynamic_Stability_Analysis/
│
├── docs/                              # 项目文档
│   ├── doc-requirements.md            # 需求文档
│   ├── doc-architecture.md            # 架构文档
│   ├── doc-plan.md                    # 方案文档
│   └── tasks/done/                    # T01-T16 任务文档
│
├── latex_report/                      # ⭐ LaTeX 独立编译包 (可上传 Overleaf)
│   ├── main.tex                       # 主文件
│   ├── sections/                      # 11 个章节 (.tex)
│   │   ├── 00_cover.tex
│   │   ├── 01_abstract.tex
│   │   ├── 02_introduction.tex
│   │   ├── 03_dataset.tex
│   │   ├── 04_surrogate.tex
│   │   ├── 05_pso.tex
│   │   ├── 06_active_learning.tex
│   │   ├── 07_multi_scenario.tex
│   │   ├── 08_porpoising.tex
│   │   ├── 09_experiments.tex
│   │   └── 10_conclusion.tex
│   └── figures/                       # 29 张被引用的图片
│
├── reports/                           # 原始报告文件 (与 latex_report/ 内容同步)
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
├── src/                               # 源代码
│   ├── preprocessing/                 # 数据清洗、归一化、特征工程
│   ├── models/                        # 6 个代理模型 + evaluate.py
│   ├── optimization/                  # PSO (base/adaptive/risk_sensitive) + GA + DE + SA + fitness
│   ├── analysis/                      # region_discovery, explainability, landscape_diagnosis
│   ├── scenarios/                     # 4 场景定义 + 多场景执行器
│   ├── visualization/                 # EDA、场景、Porpoising 风险图表
│   └── utils/                         # config.py, metrics.py
│
├── experiments/                       # 11 个实验脚本
│   ├── run_ci_comparison.py           # ⭐ GA/DE/SA 对比 (T11 新增)
│   ├── run_multi_scenario.py          # 多场景优化
│   └── ...                            # 其余 9 个脚本
│
├── figures/                           # 图表目录 (含 16_ci_comparison.png)
│
├── data/                              # 原始数据 + 处理后的数据
│   ├── raw/
│   └── processed/
│
├── outputs/                           # 训练好的模型 + 场景寻优结果
│   ├── models/                        # M1-M6 模型文件 (.pkl, .pt)
│   └── scenarios/
│
├── ppt_materials/                     # PPT 相关资料 (已完成的旧版, 不用管)
├── ppt_plan.md                        # 新版 PPT 23 页规划
├── speech_notes.md                    # Slides 14-22 答辩发言稿
├── slide15_porpoising.md              # Slide 15 详细设计文档
├── refine_advise.md ~ refine_advise6.md  # 6 轮审查意见原文
└── AGENTS.md                          # AI Agent 开发约定
```

---

## 五、Python 环境

| 项目 | 值 |
|------|-----|
| Windows Python | `/mnt/e/python313/python.exe` |
| WSL Python3 | `/usr/bin/python3` (Python 3.12.3) |
| 关键依赖 | numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, xgboost, torch, shap, joblib, python-pptx |

- Windows Python 路径含中文时可能编码失败，推荐使用 WSL Python3 运行脚本
- 所有依赖已在 WSL Python3 中安装 (如果缺少，用 `pip install --break-system-packages <包名>`)

---

## 六、报告质量状态

经过 **6 轮独立学术审查**，报告已达到以下水平:

| 维度 | 评分 (据最后一轮审查) |
|------|:---:|
| 工作量 | 97 |
| 实验完整性 | 95 |
| 计算智能体现 | 95 |
| 报告规范性 | 94 |
| 学术严谨性 | 92 |
| 独立思考 | 95 |
| **综合** | **95-97** |

**报告核心结论** (一句话):

> 在代理模型驱动的优化中，优化算法并非瓶颈——数据分布结构才是优化效果的根本约束。
> 当 86% 样本趋近饱和时，PSO、GA、DE 的结果趋于一致 (极差仅 0.15)，
> 优化景观而非优化算法决定了寻优上限。

**报告现存自反性说明** (主动承认的局限性，在 T13-T16 中专门添加):

| 章节 | 自反性内容 |
|------|------|
| §4 (代理模型) | RF=1.0 不能完全由 XGBoost 同源解释；K-Fold 不能排除同源数据偏置；XGBoost 同架构共享归纳偏置 |
| §5 (PSO) | 风险敏感有效性证据来自间接渠道；密度惩罚 60.25 可能仅因参数尺度；PSO 非"显著优于"GA/DE，而是"综合性价比最佳"；DeepEnsemble/XGBoost 角色切换需说明 |
| §6 (精炼) | 非标准 Active Learning → 不确定性引导局部重采样；仅单轮 → 非闭环；样本重复污染 → 隐式重加权 |
| §7 (多场景) | 非 NSGA-II → 加权标量化；非 Pareto 前沿 → 经验权衡前沿面；PSO 轨迹覆盖 ≠ 全空间；DRS 在数据生成器中近乎失效；权重变化未改变解 → 多目标景观仍退化 |
| §8 (SHAP) | SHAP≠因果 → 统计关联；共线性 (r=0.895) → 贡献可能被归因偏差 |
| §9 (景观诊断) | 目标函数退化 = 全篇统一解释框架；数据>优化器 |

---

## 七、LaTeX 编译说明

- `latex_report/` 是整个文件夹，**可以原封不动上传到 Overleaf**
- 编译器: **XeLaTeX**
- 文档类: `ctexart` (支持中文)
- 所有图片路径: `figures/xxx.png` (相对路径，已全部就位，共 29 张)
- 章节引用: `\input{sections/xx.tex}` (相对路径)
- 不需要 `references.bib` (本报告不含参考文献引用)

---

## 八、关键设计决策记录

以下是 6 轮修正中最重要的设计决策，后续修改**不应推翻**:

| 决策 | 理由 |
|------|------|
| "主动学习" → "不确定性引导的模型精炼" | 无新真实标签，仅有重采样 |
| "Pareto 前沿" → "经验权衡前沿面"/"标量化解分布" | 加权标量化非 NSGA-II |
| "闭环机制" → "单轮反馈精炼机制" | 仅执行了一轮 |
| "安全寻优" → "可靠性约束寻优" | σ(x) 是代理，非物理风险 |
| "核心创新" → "核心改进策略"/"重要方法设计" | μ-λσ 非新范式 |
| 场景 speed 改为固定值 (345/300/320/290) | 匹配真实赛道设计极速 |
| 场景 wing 下限统一为 20° | 真实 F1 工程约束 |
| 第 9 章 → 全篇统一解释框架 | 反向解释 §4-§7 所有现象 |
| RF=1.0 的独立解释 | 数据生成机制具有结构规律性，RF 架构不同但仍能复现 |
| SHAP 结论加"共线性限定" | downforce↔drag r=0.895，不能证因果 |

---

## 九、已完成事项 (不要再做)

| 事项 | 说明 |
|------|------|
| PPT 生成 | 23 页答辩 PPT 已定稿，`speech_notes.md` 已有完整发言稿。**不要再重新规划或生成 PPT** |
| LaTeX 报告 | 报告已历经 6 轮修正，处于定稿状态。**除非有新的审查意见，否则不要修改 tex 文件** |
| CI 对比实验 | `figures/16_ci_comparison.png` 已生成，GA/DE/SA 代码已写。**不要再重新运行** (20 trials × 4 algos 耗时约 5 分钟) |
| 6 轮审查修正 | T11-T16 全部记录在 `docs/tasks/done/`，**不要再翻出旧问题** |

---

## 十、可选的后续工作方向

以下方向在结论中列为"未来工作"，但尚未实施。如果下一个 Session 需要继续推进项目，可从以下选项中选取:

| 方向 | 说明 | 预估工作量 |
|------|------|:---:|
| **代码仓库整理** | 删除 `ppt_materials/` (旧版 PPT)、删除临时文件、更新 `README.md`、统一 `.gitignore` | 0.5h |
| **补充实验运行** | K-Fold 交叉验证脚本 (`evaluate.py` 中的 `cross_validate` 和 `leave_region_out_validate` 已编写但未独立运行) | 0.5h |
| **答辩模拟准备** | 基于 `speech_notes.md` 和 `ppt_plan.md` 排练 18 分钟答辩 | 1h |
| **物理信息代理模型** | 将气动平衡方程嵌入神经网络 (Physics-Informed Neural Network) | 2-4h |
| **贝叶斯优化对比** | 将风险敏感 PSO 与 Bayesian Optimization (利用 DeepEnsemble 的 σ(x) 做采集函数) 对比 | 2-3h |
| **CFD-代理混合策略** | 在精炼框架中引入少量真实 CFD 数据替代检索式重采样 | 3-5h |
| **生成项目总结文档** | 为课程设计归档创建最终版的 `PROJECT_SUMMARY.md` | 0.5h |

---

## 十一、新 Session 启动检查清单

新 Agent 在开始任何工作前，请按以下顺序操作:

```
1. 读取本文件 (next_session.md)
2. 读取 AGENTS.md (开发约定)
3. 读取 docs/doc-plan.md (全局方案)
4. 确认 Python 环境可用: python3 --version
5. 确认报告状态: 检查 latex_report/sections/ 目录
6. 明确本 Session 的目标任务
7. 创建 docs/tasks/todo/T17_xxx.md 任务文档
8. 开始工作
```

---

**文档生成时间**: 2026-06-14  
**当前会话任务**: T01-T16 全部完成、PPT 完成、发言稿完成  
**下一会话任务**: 待定 
