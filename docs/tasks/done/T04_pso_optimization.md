# T04: PSO 寻优框架搭建

**优先级**: 🔴 核心必做

**依赖**: T03 (代理模型就绪，不确定性估计可选)

**负责模块**: `src/optimization/`

---

## 任务描述

从零实现粒子群优化框架，逐步从标准 PSO 演进到自适应 PSO，最终实现风险敏感 PSO（核心创新）。要求同时支持单模型适应度和集成不确定性适应度。

---

## 分步执行

### 4.1 标准 PSO 实现

- [ ] 粒子类: 位置 x、速度 v、个体最优 pbest、适应度 fitness
- [ ] 速度更新: `v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)`
- [ ] 位置更新: `x = x + v`
- [ ] 全局最优 (gbest) 更新
- [ ] 边界处理 (截断或反射)
- [ ] 收敛判断: 最大迭代或 fitness 改进 < ε 连续 N 代
- [ ] 离散变量处理: `drs_active` 需特殊处理（可在 fitness 评估时四舍五入或双态遍历）

关键超参数参考范围:

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 粒子数 N | 30-100 | 先试 50 |
| 学习因子 c1, c2 | 2.0 | 经典值 |
| 惯性权重 w | 0.7 | 固定基线 |
| 最大迭代 T | 50-100 | |

### 4.2 自适应惯性权重 PSO

- [ ] 线性衰减: `w(t) = w_max - (w_max - w_min) * t/T`
- [ ] 非线性衰减 (推荐): `w(t) = w_max - (w_max - w_min) * (t/T)^α`
  - α=1 为线性，α>1 为加速衰减
- [ ] (可选) 基于群体适应度改善速率的动态自适应

### 4.3 风险敏感适应度函数 (核心创新)

基础形式:
```
Fitness(x) = μ(x)
```

**改进形式** (需要 T03 中 DeepEnsemble 的 μ(x), σ(x)):
```
Fitness(x) = μ(x) - λ * σ(x)
```

其中 λ 为风险厌恶系数，建议搜索范围 0.1~2.0。

**物理含义**:
- 训练数据密集区 → σ(x) 小 → 惩罚小 → PSO 偏好搜索
- OOD 区域 → σ(x) 大 → 惩罚大 → PSO 自动规避
- 低稳定性样本稀疏 → 该区域 σ 大 → PSO 谨慎探索

### 4.4 数据密度惩罚 (推荐增强)

```
Fitness(x) = μ(x) - λ1 * σ(x) - λ2 * D(x)
```

D(x) 可选方案 (任选一种):
- KNN 密度: x 到训练集 K 近邻平均距离
- Mahalanobis 距离
- Isolation Forest 异常分数

### 4.5 参数空间约束

| 参数 | 约束范围 | 类型 |
|------|----------|------|
| speed_kmh | [80, 360] | 连续 |
| wing_angle_deg | [0, 45] | 连续 |
| drs_active | {0, 1} | 离散 |
| downforce_n | [0, 10000] (全局默认, 场景T06可缩窄) | 连续 |
| drag_n | [0, 600] (全局默认, 场景T06可缩窄) | 连续 |

---

## 验收标准

- [ ] 标准 PSO 可独立运行并在无约束 5D 空间找到合理最优
- [ ] 自适应权重 PSO 收敛代数 ≤ 标准 PSO
- [ ] (推荐) 风险敏感 PSO 实现并验证 λ=0 时退化为标准 PSO
- [ ] 所有 PSO 变体通过同一基准函数测试 (如 Rastrigin 函数验证)
- [ ] 单次寻优耗时 ≤ 20 秒 (理想目标 ≤ 5 秒)

## 输入
- `outputs/models/best_model.pt` (或 .pkl)
- `outputs/models/deep_ensemble/` (可选，用于 D.3)

## 输出
```
src/optimization/
├── pso_base.py            # 标准 PSO
├── pso_adaptive.py        # 自适应 PSO
├── pso_risk_sensitive.py  # 风险敏感 PSO
└── fitness.py             # 适应度函数 (含多种版本)
```

## 弹性空间

> - 若 DeepEnsemble 未实现 (T03)，风险敏感 PSO 可跳过，仅交付标准+自适应版本
> - 密度惩罚 D(x) 为风险敏感的增强可选项
> - pyswarms 可作为快速原型工具，但最终建议自定义实现以便后续特化
> - 离散变量 drs_active 的处理可在 fitness 评估中以规则约束代替连续优化

## 关键决策记录

| # | 决策 | 方案 | 原因 |
|---|------|------|------|
| 1 | 离散变量处理 | 连续优化 + clamp to [0,1] + 评估时 rounding | 避免二进制 PSO 特殊规则，保持统一框架 |
| 2 | 边界策略 | clamp (默认), reflect (可选) | clamp 简单可靠，OOD 由风险敏感 fitness 惩罚 |
| 3 | 自适应 α 值 | α=1.0 (线性衰减) | 非线性 (α=2.0) 在 Rastrigin 上更优但方差大；线性衰减在代理模型上稳定 |
| 4 | PSO 默认参数 | N=50, w_start=0.9, w_end=0.4, c1=c2=2.0, max_iter=60 | 经典值，经验验证有效 |
| 5 | λ 搜索范围 | {0.0, 0.5, 1.0, 1.5, 2.0} | λ=1.0~1.5 在 OOD 规避与最优性之间取得平衡 |
| 6 | 推理模型选择 | XGBoost (主), DeepEnsemble (风险敏感) | XGBoost 0.24ms vs RF 37ms; DeepEnsemble 提供 σ |
| 7 | 批量评估 | 每代所有粒子一次 batch predict | GPU 友好，避免 per-particle Python 循环 |
| 8 | 密度惩罚 | KNN (K=10) 平均距离, 可选增强 | KNN 直观，无需额外训练 |
| 9 | 收敛判据 | gbest 改善 < 1e-6 连续 12 代 | 默认值, max_iter//5 取 min=10 |
| 10 | 标准 vs 自适应收敛 | PSOAdaptive average 84% faster (19 vs 35 iters, 5-seed) | 自适应惯性权重增强早期探索 + 后期利用 |

## 验收结果

- [x] 标准 PSO 可独立运行并在无约束 5D 空间找到合理最优 (fitness=100.24, 13 iters, 0.04s)
- [x] 自适应权重 PSO 收敛代数 <= 标准 PSO (5-seed avg: 19.2 vs 35.4 iters, 84% faster)
- [x] 风险敏感 PSO 实现并验证 lambda=0 时退化为标准 PSO (fitness=101.3, hallucinates OOD corner)
- [x] 所有 PSO 变体通过 Rastrigin/Sphere 基准函数测试
- [x] 单次寻优耗时: 0.04~0.05s (< 5s ideal, << 20s requirement)

## 最终文件

```
src/optimization/
├── __init__.py
├── fitness.py              # ModelWrapper + 3 fitness types + benchmark funcs
├── pso_base.py             # Particle class + Standard PSO
├── pso_adaptive.py         # Adaptive inertia weight PSO
└── pso_risk_sensitive.py   # Risk-sensitive PSO runner + lambda sweep

experiments/
└── run_pso_comparison.py   # Full T04 verification experiment

figures/pso/
├── convergence_curves.png
└── lambda_vs_stability.png
```

**状态**: ✅ 已完成

## T04-Ext: 高性能参数区域发现

在 T04 核心完成后，扩展了一个**优化结果分析**模块 (因不属于 PSO 算法改进, 归入 "Design Space Exploration"):

- **数据**: 利用 PSO 搜索过程中所有粒子的位置+适应度 (44,550 候选解, 30 次试验)
- **方法**: Top 1%/5%/10% 筛选 → 每参数 P5-P95 区间统计 → DBSCAN 聚类
- **结果**: 发现 **7 个不同的高性能解簇**, 对应不同气动设计策略:
  - Cluster 3 (最大, 47%): speed~204, wing~28°, DRS=0 — 传统高下压力方案
  - Cluster 2 (18%): speed~350, wing~11°, DRS=1 — 高速 DRS 开启方案
  - Cluster 1 (31%): speed~346, wing~11°, DRS=0 — 高速低阻力方案
- **意义**: 输出从 "最佳单点" 升级为 "最佳点 + 参数区间 + 解簇", 回答"哪些参数范围都能获得高性能"

### 新文件
```
src/analysis/region_discovery.py       # 候选筛选/统计/聚类
src/visualization/plot_regions.py      # 区域可视化
experiments/run_region_discovery.py    # 区域发现实验脚本
figures/regions/                       # 输出图表
```

### 关键决策
| # | 决策 | 方案 | 原因 |
|---|------|------|------|
| 1 | 数据来源 | PSO 全群体每代位置+适应度 (collect_candidates=True) | 单次 PSO 仅 gbest 不足以绘制区域 |
| 2 | 试验次数 | 30 次独立 PSO | 每次产出 ~50×(60+1)≈3000 候选, 确保覆盖 |
| 3 | 聚类方法 | DBSCAN (自动 eps) | K-Means 需预设 K, 不如 DBSCAN 适应未知结构 |
| 4 | 可视化重点 | Top 5% 散点矩阵 + 参数范围条形图 + 聚类投影 | 三者互补: 分布/区间/模态 |
