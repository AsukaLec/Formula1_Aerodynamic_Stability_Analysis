# T06: 多场景优化与可解释性分析

**优先级**: 🔴 核心必做

**依赖**: T04 (PSO 框架就绪), T03 (代理模型就绪)
**前置安装**: SHAP 已在主机 Python 环境中安装 (`/mnt/e/python313/python.exe -m pip show shap`)，无需重复安装

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

- [x] 对每个场景，加载对应约束，运行 PSO
- [x] 记录最优解 (x*, f*)、收敛代数、收敛曲线
- [x] 每个场景独立运行 ≥ 10 次，统计最优解的均值与方差 (实际: 15次×4场景)
- [x] 生成 `outputs/scenarios/` 下各场景结果

### 6.3 可解释性分析

#### 6.3.1 SHAP 分析 (推荐)
- [x] 全局 SHAP: 特征重要性排序 (summary plot)
- [x] 局部 SHAP: 对每场景最优解生成 waterfall plot (S1/S2/S3/S4 共4张，diverging bar + symlog)
- [~] dependence plot: speed 和 wing_angle 对 stability 的非线性影响曲线 (跳过 — 非验收标准强制项)

#### 6.3.2 场景对比雷达图 (推荐)
- [x] 将各场景最优解特征值归一化到全局PSO边界 [0,1] (非per-column min/max)
- [x] 多场景雷达图展示参数倾向差异 (4场景 + 无约束基准虚线)

#### 6.3.3 约束灵敏度分析 (推荐)
- [x] 微调场景约束 ±10%，观察最优解和适应度变化
- [x] 龙卷风图 (Tornado Plot) 展示各约束灵敏度排序 (S1/S2/S3/S4 共4张, 双柱式)

#### 6.3.4 反事实分析 (可选进阶)
- [x] 搜索"使稳定性跌破阈值所需的最小参数扰动"
- [x] 量化最优解的鲁棒性边界 (结论: 数据不平衡导致无法找到跌破95的有效扰动)

#### 6.3.5 决策规则提炼 (可选进阶)
- [x] 在高稳定性区域训练浅层决策树
- [x] 提取可读规则 (depth=3, drag_n和speed_kmh为主要分裂特征)
- [x] 与 PSO 最优解对照验证

### 6.4 寻优轨迹可视化 (可选)
- [x] PCA 降维至 2D (S1+S3联合PCA, PC1=65.9%, PC2=14.7%, 合计80.5%)
- [x] 绘制粒子多代迁移路径 (top-5粒子 + gbest轨迹 + hexbin背景)
- [x] 标注各场景最优解位置 (金色星形 + 末代适应度标注)

---

## 验收标准

- [x] 至少完成 S1 + S2 两组场景的寻优并输出最优参数组合 (实际: S1+S2+S3+S4 四组)
- [x] 每场景独立运行 ≥ 10 次，记录均值与方差 (实际: 15次/场景)
- [x] 完成全局 SHAP 分析 + 至少 2 个局部 SHAP waterfall (实际: 4个)
- [x] 生成至少 1 张多场景雷达图
- [x] (推荐) 完成约束灵敏度龙卷风图 (实际: 4张)
- [x] (推荐) 反事实分析完成
- [x] (推荐) 决策规则提炼完成

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

### 1. OOD策略选择
采用方案A（按任务定义边界+λσ惩罚），PSO在wing_angle=0(S1)和drag=0(S2)的OOD边缘发现高预测值，表明λ=1.0的惩罚在数据极端不平衡（85%稳定性>99）下不足以完全抑制OOD幻觉。T08消融实验中λ=0 vs λ=2.0的对比将验证这一点。

### 2. SHAP目标模型
选择XGBoost+TreeExplainer（非DeepEnsemble），原因：①TreeExplainer精确且快 ②文档明确支持NN回退到Permutation ③全局SHAP结论与Permutation Importance一致（downforce_n排名第一）。

### 3. 各场景最优解

| 参数 | S1 Monza | S2 Monaco | S3 Balanced | S4 Wet |
|------|----------|-----------|-------------|--------|
| speed_kmh | 280.0 (下界) | ~158.1 | ~204 | ~163 |
| wing_angle_deg | 0.0 (下界/OOD) | 20.0 (下界) | ~10.2 | 15.0 (下界) |
| drs_active | 1.0 (固定) | 0.0 (固定) | 0.0 | 0.0 (固定) |
| downforce_n | ~3210 | 3000.0 (下界) | ~745 | ~903 |
| drag_n | ~44.5 | 0.0 (下界/OOD) | ~122 | ~73 |
| stability | ~99.74 | ~99.79 | ~99.83 | ~99.82 |
| unconstrained baseline | — | — | — | 99.83 |

S3和S4有明显更大的variation（downforce_n std~300-450），因为它们没有硬性约束推到边界。

核心发现：两场景最优解均推到约束下界，downforce_n和drag_n倾向极小化。
这与SHAP结论一致：downforce_n是绝对主导特征(mean|SHAP|=1.88)，其他特征贡献<0.002。

### 4. 约束灵敏度

- **S1**: speed_lb最敏感(±0.023)，放开速度下限可显著提升适应性
- **S2**: wing_lb(±0.011)和downforce_lb(±0.009)中等敏感，speed_ub不敏感
- **S3**: wing_lb, speed_ub, wing_ub三者灵敏度接近(均~±0.014)，场景约束平衡时灵敏度均匀
- **S4**: wing_ub最敏感(-0.011)，增大翼角上限会降低适应度

### 5. 反事实分析
搜索500步内未能找到使稳定性跌破95的最小扰动，原因：XGBoost在参数空间内预测值集中在99-100区间（数据不平衡的直接后果）。这表明单纯依赖预测值做反事实分析不适用，需结合不确定性σ。

### 6. 决策规则
drag_n和speed_kmh是高层可解释规则中的主要分裂特征，提取了depth=3的浅层决策树。

### 7. 验收达成
- ✅ S1+S2+S3+S4 四组场景各≥15次独立运行
- ✅ 全局SHAP + 4个局部waterfall (S1/S2/S3/S4)
- ✅ 多场景雷达图（全局边界归一化 + 无约束基准线 + 4场景）
- ✅ 约束灵敏度龙卷风图（S1/S2/S3/S4共4张）
- ✅ 反事实分析（可选，完成）
- ✅ 决策规则提炼（可选，完成）
- ✅ 所有可视化图表修正：龙卷风左右布局、SHAP/PI对数刻度、waterfall聚焦缩放
- ✅ PCA粒子迁移轨迹 (可选扩展, S1+S3, PC1=65.9% PC2=14.7% 合计80.5%)

### 8. PCA轨迹关键发现
- **PC1 (65.9%)**: 主导特征为 wing_angle_deg (0.512) — 区分高/低翼角场景
- **PC2 (14.7%)**: 主导特征为 drs_active (0.926) — 区分 DRS=0 vs DRS=1 两簇
- **S1 轨迹**: 粒子从随机散布快速集中 → wing_angle→0, drs→1 固定方向压缩
- **S3 轨迹**: 粒子搜索更分散，多样性高于 S1 (约束更宽松)
