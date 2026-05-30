# T06: 多场景优化与可解释性分析

**优先级**: 🔴 核心必做

**依赖**: T04 (PSO 框架就绪), T03 (代理模型就绪)

**负责模块**: `src/scenarios/`

---

## 任务描述

定义多组典型 F1 赛道工况，在各自约束空间内运行 PSO 寻优，输出各场景最优参数组合。同步进行可解释性分析——不仅给出最优解，还要解释"为什么这是最优"。

---

## 分步执行

### 6.1 场景定义

必做 (至少 2 组):

| 编号 | 场景 | 速度约束 | 翼角约束 | DRS | 额外约束 |
|------|------|----------|----------|-----|----------|
| S1 | 高速赛道 (Monza) | v > 280 | [0, 15] | 1 | minimze drag |
| S2 | 高下压力 (Monaco) | v < 200 | [20, 40] | 0 | downforce > 3000N |

可选扩展:
| S3 | 中速均衡 | 200 < v < 280 | [10, 30] | free | — |
| S4 | 雨战/湿地 | v < 250 | [15, 35] | 0 | stability 优先级最高 |
| S5 | 自定义 | — | — | — | — |

### 6.2 寻优执行

- [ ] 对每个场景，加载对应约束，运行 PSO
- [ ] 记录最优解 (x*, f*)、收敛代数、收敛曲线
- [ ] 每个场景独立运行 ≥ 10 次，统计最优解的均值与方差
- [ ] 生成 `outputs/scenarios/` 下各场景结果

### 6.3 可解释性分析

#### 6.3.1 SHAP 分析 (推荐)
- [ ] 全局 SHAP: 特征重要性排序 (summary plot)
- [ ] 局部 SHAP: 对每场景最优解生成 waterfall plot，展示各特征如何贡献到最终稳定性
- [ ] dependence plot: speed 和 wing_angle 对 stability 的非线性影响曲线

#### 6.3.2 场景对比雷达图 (推荐)
- [ ] 将各场景最优解特征值归一化到 [0,1]
- [ ] 多场景雷达图展示参数倾向差异

#### 6.3.3 约束灵敏度分析 (推荐)
- [ ] 微调场景约束 ±10%，观察最优解和适应度变化
- [ ] 龙卷风图 (Tornado Plot) 展示各约束灵敏度排序

#### 6.3.4 反事实分析 (可选进阶)
- [ ] 搜索"使稳定性跌破阈值所需的最小参数扰动"
- [ ] 量化最优解的鲁棒性边界

#### 6.3.5 决策规则提炼 (可选进阶)
- [ ] 在高稳定性区域训练浅层决策树
- [ ] 提取可读规则 (如 `IF speed < X AND wing_angle > Y THEN stability > 90`)
- [ ] 与 PSO 最优解对照验证

### 6.4 寻优轨迹可视化 (可选)
- [ ] PCA/t-SNE 降维至 2D
- [ ] 绘制粒子多代迁移路径
- [ ] 标注各场景最优解位置

---

## 验收标准

- [ ] 至少完成 S1 + S2 两组场景的寻优并输出最优参数组合
- [ ] 每场景独立运行 ≥ 10 次，记录均值与方差
- [ ] 完成全局 SHAP 分析 + 至少 2 个局部 SHAP waterfall
- [ ] 生成至少 1 张多场景雷达图
- [ ] (推荐) 完成约束灵敏度龙卷风图

## 输入
- `outputs/models/best_model.pt`
- `src/optimization/pso_risk_sensitive.py`

## 输出
```
outputs/scenarios/
├── S1_monza/
│   ├── best_solution.json
│   └── convergence.csv
├── S2_monaco/
│   ├── best_solution.json
│   └── convergence.csv
└── scenario_comparison.csv

figures/scenarios/
├── scenario_radar.png
├── shap_summary.png
├── shap_waterfall_S1.png
├── constraint_sensitivity.png
└── decision_tree_rules.png  (可选)
```

## 弹性空间

> - 场景数量: 核心完成 S1+S2 即可，S3-S5 按时间扩展
> - 反事实分析和决策规则提炼为进阶项，非必须
> - SHAP 分析若模型为 XGBoost，可用内置 feature_importance 替代
> - 若代理模型为 TabNet，SHAP 支持较差，可改用 Permutation Importance

## 关键决策记录

> (记录各场景最优解、SHAP 发现的关键洞察、不同场景下特征重要性排名变化等)
