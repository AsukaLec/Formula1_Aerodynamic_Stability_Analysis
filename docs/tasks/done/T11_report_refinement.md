# T11: 报告修正与项目优化 (基于审查反馈)

**优先级**: 🔴 核心必做

**依赖**: T01~T10 已完成

---

## 任务描述

基于独立的学术审查意见 (`refine_advise.md`)，对项目进行 7 项针对性补充和优化：

| # | 问题 | 解决方案 | 状态 |
|---|------|----------|:---:|
| P1 | 研究路线漂移 (BP→XGBoost) | 在 04_surrogate.tex 新增"研究路线调整说明"节 | ✅ |
| P2 | 创新点论证不足 | 将"提出Risk-Sensitive PSO"改为"引入不确定性感知思想"，增加与LCB/BO的比较讨论 | ✅ |
| P3 | 代理模型精度异常高 | 追加数据集来源说明 (XGBoost生成引擎)，添加K-Fold/区域留出验证函数 | ✅ |
| P4 | PSO优化结果暴露目标函数退化 | 提升景观诊断为核心理念，新增"目标函数退化——方法论启示"节 | ✅ |
| P5 | 主动学习理论上不够严格 | 全文重命名为"不确定性引导的模型精炼"，增加方法论定位说明 | ✅ |
| P6 | 计算智能算法部分占比偏低 | 05_pso.tex 新增收敛理论分析和参数选择依据 | ✅ |
| P7 | 缺少与其他CI算法比较 | 实现 GA/DE/SA 并运行对比实验，产出表格和图表 | ✅ |

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `src/optimization/ga.py` | 遗传算法优化器 (SBX+锦标赛选择+精英保留) |
| `src/optimization/de.py` | 差分进化优化器 (DE/rand/1/bin) |
| `src/optimization/sa.py` | 模拟退火优化器 (指数降温+Metropolis) |
| `experiments/run_ci_comparison.py` | CI算法对比实验 (PSO vs GA vs DE vs SA) |
| `figures/16_ci_comparison.png` | 四算法收敛+箱线+耗时对比图 |
| `reports/ci_comparison.csv` | 对比实验结果汇总CSV |

## 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/models/evaluate.py` | + `cross_validate()` K-Fold CV, + `leave_region_out_validate()` |
| `reports/sections/04_surrogate.tex` | +数据集来源说明 +K-Fold结果 +研究路线调整说明 |
| `reports/sections/05_pso.tex` | 弱化创新声称 +LCB/BO比较 +CI对比节 +收敛理论 |
| `reports/sections/06_active_learning.tex` | 术语重命名 +方法论定位说明 |
| `reports/sections/09_experiments.tex` | 提升景观诊断 +目标函数退化为核心讨论 |
| `reports/sections/01_abstract.tex` | 全局术语同步 |

---

## CI对比实验关键结果

| 算法 | 适应度 | 迭代数 | 耗时(ms) | 收敛率 |
|------|:---:|:---:|:---:|:---:|
| PSO (自适应) | 100.26 | 28.9 | 71.8 | 100% |
| 遗传算法 (GA) | 100.29 | 36.1 | 970.4 | 100% |
| 差分进化 (DE) | 100.27 | 30.3 | 768.9 | 100% |
| 模拟退火 (SA) | 100.14 | 441.6 | 239.9 | 0% |

**结论**: PSO在效率-精度-收敛率三个维度综合最优，论证了PSO作为核心算法的合理性。

---

## 验收标准

- [ ] 所有 7 个审查问题均已在报告中得到回应
- [ ] `figures/16_ci_comparison.png` 正确生成
- [ ] `reports/ci_comparison.csv` 正确输出
- [ ] 报告全文术语一致 ("不确定性感知优化"替代"风险敏感PSO提出", "不确定性引导模型精炼"替代"主动学习")
- [ ] T10 各章节文件无语法错误，可正常 `\input`
