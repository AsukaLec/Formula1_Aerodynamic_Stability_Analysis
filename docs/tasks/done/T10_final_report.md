# T10: 结题报告撰写与 LaTeX 排版

**优先级**: 🔴 核心必做

**依赖**: 所有前置任务完成 (T01~T09)

---

## 任务描述

汇总全部实验数据、图表和分析结论，以 `format_example.tex` 为模板，生成 LaTeX 格式的《计算智能》课程设计结题报告。最终在 Overleaf 上使用 XeLaTeX 编译为 PDF。

---

## 分步执行

### 10.1 报告大纲

对齐 `format_example.tex` 体例（`ctexart` 文档类），章节结构如下：

1. **摘要** — 问题、方法、关键结论 (200-300 字)，含关键词
2. **研究背景** — 问题定义、研究目标（含核心公式）、研究意义
3. **数据分析与预处理** — 数据来源、数据清洗、EDA、不平衡分析
4. **代理模型构建** — 候选模型描述、训练配置、对比实验、模型选择
5. **粒子群优化算法** — 标准 PSO 公式、自适应改进、风险敏感适应度（创新点 ①）
6. **主动学习迭代精炼** — (若 T05 完成) 闭环机制与效果
7. **多场景优化与可解释性** — 场景定义、寻优结果、SHAP/雷达图/灵敏度分析
8. **Porpoising 风险分析** — 一阶梯度风险热力图 + 二阶 Hessian 曲率分析
9. **实验结果与分析** — 对比实验、消融实验、鲁棒性分析
10. **结论与展望** — 主要贡献总结、不足与未来工作

### 10.2 量化指标汇总

| 指标 | 实际值 | 达标/理想 |
|------|--------|-----------|
| 代理模型 R² | 0.9999 (XGBoost) | ✅ 达标 / ✅ 达理想 |
| 代理模型 MSE | 0.0312 (XGBoost) | ✅ 达标 / ≈ 达理想 |
| PSO 收敛代数 | 27.9 (标准) / 22.1 (自适应) | ✅ 达标 / ✅ 达理想 |
| 最优解标准差 | 0.0836 (20次) | 接近达标 (受 σ 惩罚影响) |
| 单次寻优耗时 | 83ms | ✅ 达标 / ✅ 达理想 |

### 10.3 成果汇总

- [x] GitHub 代码仓库整理完毕，README 完善
- [x] `reports/main.tex` 及所有章节文件内容完整 (11 个 .tex 文件)
- [x] `reports/references.bib` — 用户决策：无需参考文献条目
- [x] 检查全文无 "TODO"、"待完成" 等遗留标记

### 10.4 LaTeX 文件生成

- [x] 以项目根目录 `format_example.tex` 为模板创建 `reports/main.tex`
  - 导言区（\documentclass + 宏包）与模板保持一致
  - 封面（titlepage）：学校、学院、课程名称、课题标题、姓名学号
  - 摘要 + 关键词
  - 目录（\tableofcontents）
  - 各章节通过 `\input{sections/xxx.tex}` 引入

- [x] 文件结构:
  ```
  reports/
  ├── main.tex                    # 主文件
  ├── sections/
  │   ├── 01_abstract.tex         # 摘要 (中/英, 含关键词)
  │   ├── 02_introduction.tex     # 研究背景
  │   ├── 03_dataset.tex          # 数据分析与预处理 (T01/T02)
  │   ├── 04_surrogate.tex        # 代理模型构建 (T03)
  │   ├── 05_pso.tex              # 粒子群优化算法 (T04)
  │   ├── 06_active_learning.tex  # 主动学习迭代精炼 (T05, 若实现)
  │   ├── 07_multi_scenario.tex   # 多场景优化与可解释性 (T06)
  │   ├── 08_porpoising.tex       # Porpoising 风险分析 (T07, ✅)
  │   ├── 09_experiments.tex      # 实验结果与分析 (T08)
  │   └── 10_conclusion.tex       # 结论与展望
  └── references.bib              # BibTeX 参考文献库 (未创建—用户决策无需参考文献)
  ```

- [x] 排版约定 (对齐模板):
  - 表格: `booktabs` 三线表风格（\toprule / \midrule / \bottomrule）
  - 公式: `equation` 或 `align` 环境
  - 图表: `\includegraphics{figures/xxx.png}`，使用 `figure[H]` 浮动
  - 代码: `lstlisting` 环境（Python 代码片段）

- [x] Overleaf 编译:
  - 编译器: **XeLaTeX**
  - 上传 `reports/` 整个目录 + `figures/` 目录
  - 保持相对路径不变（`\includegraphics{figures/...}` 引用）

---

| 验收标准 |
|---------|
| [x] `reports/` 下所有 `.tex` 文件书写完整，无 TODO 占位符 |
| [ ] 可在 Overleaf 上用 XeLaTeX 成功编译为 PDF |
| [x] 报告覆盖全部核心模块 (T01~T09) 的实验结果 |
| [x] 量化指标表填写完整 |
| [ ] 至少 1 位组员通读全文 |

## 输入
- `format_example.tex` (模板)
- `figures/` 目录 (所有图表)
- `reports/*.csv` / `reports/*.md` (各模块实验数据)

## 输出
- `reports/main.tex` (主文件)
- `reports/sections/01_abstract.tex` ~ `10_conclusion.tex` (各章节)
- `reports/figures/` (28 张引用图片)

### 关键决策记录

| 决策 | 内容 |
|------|------|
| 参考文献 | 用户决策：无需参考文献条目，`references.bib` 不创建 |
| T05 章节 | 作为独立第6章写入报告 |
| 多目标版本 | 以 v2 多目标为主，同时对比 v1 单目标 |
| Landscape Diagnosis | 在9.6节作为核心景观诊断证据链呈现 |
| Porpoising 章节 | 独立第8章（一阶梯度 + 二阶Hessian） |
| 图片路径 | 所有图片复制到 `reports/figures/`，LaTeX 中 `figures/xxx.png` 相对引用 |
| Overleaf 部署 | 打包 `reports/` 目录上传，XeLaTeX 编译 |

### 完成统计

| 项目 | 数量 |
|------|:---:|
| 主文件 | 1 (`main.tex`) |
| 章节文件 | 11 (含 00_cover.tex) |
| 引用图片 | 28 |
| 总字数 | ~6000 字 |
| 报告章节 | 10 章 |

## 弹性空间

> - T05 主动学习已实现 (浅层一轮精炼)，作为独立第6章写入报告
> - 最优解标准差接近达标 (0.0836)，受 σ 惩罚和模型饱和影响，在报告中如实分析
> - 参考文献由用户决策省略，主文件中无 thebibliography 环境
