# T05: 主动学习迭代精炼

**优先级**: 🟢 可选 (创新加分项)

**依赖**: T03 (代理模型 + DeepEnsemble), T04 (风险敏感 PSO 可运行)

**负责模块**: `src/optimization/active_learning.py`

---

## 任务描述

在 PSO 寻优过程中，利用不确定性估计识别代理模型不可信的区域，针对性增补数据并增量训练代理模型，形成 "PSO 探索 → 不确定性反馈 → 模型精炼 → 更准确的寻优" 闭环。

---

## 分步执行

### 5.1 浅层方案 (一轮迭代)

- [x] PSO 运行完毕后，收集搜索过程中 σ(x) 最高的 Top-K 候选点
- [x] 从原始数据集中查找这些点的最近邻真实样本
- [x] 将最近邻样本加入训练集，重新训练一次代理模型
- [x] 对比精炼前后模型在测试集上的指标变化

### 5.2 中层方案 (多轮迭代)

- [x] 每 N 代 PSO 迭代后 (N=10)，触发一次主动学习
- [x] 选择 σ(x) 大且 μ(x) 高的候选点（不确定性高但有潜力）
- [x] 增量更新代理模型 (fine-tune 而非从头训练)
- [x] 记录每轮精炼后的模型精度和 PSO 收敛行为变化

### 5.3 深层方案 (可选)

- [ ] 使用 Expected Improvement (EI) 或 Upper Confidence Bound (UCB) 采集函数 (可选, UCB 方法已实现在 UncertaintySampler.ucb(), 待集成到 PSO 初始化)
- [ ] 将采集函数作为粒子初始化偏置，引导 PSO 优先探索高信息增益区域 (可选)

---

## 验收标准

- [x] 至少实现浅层方案，并有精炼前后模型精度对比
- [x] 若实现中层方案: 展示 PSO 收敛曲线在精炼前后的变化
- [x] 报告中能清晰描述主动学习闭环的工作机制

## 输入
- `outputs/models/deep_ensemble/`
- T04 的 PSO 运行日志

## 输出
```
outputs/models/
├── model_refined.pt
reports/active_learning_report.md
figures/pso/
└── active_learning_convergence.png
```

## 弹性空间

> - 此为高阶创新模块，完全可选
> - 此为高阶创新模块，完全可选。推荐至少实现浅层方案以体现创新
> - 精力和算力允许时实施中层方案，用图表展示闭环效果
> - 不实现不影响整体课题完成度，成功实施则显著提升创新性

## 关键决策记录

| # | 决策 | 方案 | 原因 |
|---|------|------|------|
| 1 | 浅层重训策略 | 从零训练新 DeepEnsemble（非加载 pretrained 后继续训练） | 更干净的对比基准，避免预训练权重污染 |
| 2 | 中层初始模型 | 直接加载 T03 预训练模型，Round 0 不额外训练 | 保持与 baseline 一致的起点 |
| 3 | 中层增量策略 | fine_tune (低 lr=5e-4, 少 epoch=30) 而非全量重训 | 快速迭代，避免灾难性遗忘 |
| 4 | y 尺度处理 | 训练时 y/100，评估时 pred*100 | 与 T03 模型训练流程一致（MLP 在 [0,1] 区间更稳定） |
| 5 | 最近邻查找 | 在 MinMax 缩放空间用 sklearn NearestNeighbors | 与模型特征空间一致，最近邻含义准确 |
| 6 | 采样策略 | 浅层: top_sigma; 中层: promising (α=0.5) | 浅层聚焦最高不确定性；中层平衡潜力与不确定性 |
| 7 | 去重处理 | 中层多轮间维护已添加索引集合，仅添加新样本 | 避免反复添加相同样本导致数据倾斜 |
| 8 | 代码输出语言 | 所有 print/日志使用英文 | 遵循 AGENTS.md 约定，避免 GBK 编码问题 |

## 验收结果

- [x] 浅层方案: PSO -> top-K高σ候选 -> 最近邻 -> 重新训练 -> 对比
  - R2: 0.9995 -> 0.9998 (+0.0003)
  - MSE: 0.3272 -> 0.1444 (-55.9%)
  - PSO 收敛对比: 精炼后收敛速度快 2.1x (18 vs 37 代), gbest 更保守 (100.02 vs 101.30, 避免 OOD 幻觉)
- [x] 中层方案: 3轮 PSO+采样+微调循环
  - Round 0: R2=0.9995, MSE=0.3272
  - Round 1: R2=0.9998, MSE=0.1259
  - Round 2: R2=0.9998, MSE=0.1133
  - Round 3: R2=0.9999, MSE=0.0903
  - MSE 总降幅: -72.4%
- [x] 报告清晰描述闭环工作机制
- [x] 不破坏已有 T01-T04 代码

## 最终文件

```
src/optimization/
├── active_learning.py      # UncertaintySampler + ActiveLearner (shallow + medium)

src/models/
├── deep_ensemble.py        # 新增 fine_tune() 增量训练方法

src/utils/
├── config.py               # 新增 AL 配置常量

experiments/
├── run_active_learning.py  # T05 完整实验脚本

outputs/models/
├── deep_ensemble_refined/  # 精炼后模型权重

reports/
├── active_learning_report.md

figures/active_learning/
├── uncertainty_distribution.png
├── active_learning_convergence.png
├── pso_convergence_comparison.png
```

**状态**: ✅ 已完成
