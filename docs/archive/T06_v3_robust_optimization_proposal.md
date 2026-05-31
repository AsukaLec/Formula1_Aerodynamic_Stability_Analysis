# T06 多目标扩展 — 问题诊断与解决方案

> 生成时间：2026-05-31  
> 状态：待评估

---

## 1. 当前问题全景

### 1.1 发生了什么

T06 v1（单目标最大化 stability_index）4 场景最优解全部拥挤在 99.74~99.83，雷达图呈对称六边形，PSO 搜索能力无法体现。

T06 v2（多目标加权 stability + efficiency + power）将问题从单目标扩展为三维 Pareto，新增了 7 张分析图。4 场景复合 fitness 出现了有意义的区分（S1=0.685 vs S2=0.894），但**所有 4 个场景的 wing 全部收敛至下界 20°**，PSO 没有做出不同场景的不同翼角权衡。

### 1.2 根因链

```
合成数据集 (speed, wing, downforce, drag 独立均匀生成)
    ↓
stability_index ≈ 100 为绝对主导类 (85% 样本 >99)
    ↓ (已做样本加权 28.6:1，但)
5 个输入与 stability 之间缺乏条件依赖关系
    ↓
XGBoost 仅依赖 downforce_n 一个特征 (mean|SHAP|=12.2, 其余<0.3)
DeepEnsemble 对所有中高 downforce 输入预测 stability≈100
    ↓
wing_angle 在模型输出中 "不可见" — 无论 wing=20° 还是 35°，预测无差异
    ↓
任何目标公式只要包含 stability，就会将 PSO 推向 wing 下界 (drag 也推向下界)
    ↓
多目标 tradeoff 无法在 wing 维度建立真正的张力
```

---

## 2. v2 多目标尝试的具体结果

### 2.1 4 场景多目标最优解

| 场景 | speed | wing | drs | downforce | drag | fitness | stability | efficiency (N/N) | power (norm) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| S1 Monza (w: 0.4/0.3/0.3) | 345 | **20.0°** | 1 | 4249 | 18.0 | 0.685 | 99.8 | 236 | 0.040 |
| S2 Monaco (w: 0.5/0.4/0.1) | 300 | **20.0°** | 0 | 4454 | 18.0 | 0.894 | 99.8 | 248 | 0.035 |
| S3 Balanced (w: 0.4/0.4/0.2) | 320 | **20.0°** | 1 | 4412 | 18.0 | 0.790 | 99.8 | 245 | 0.037 |
| S4 Wet (w: 0.6/0.2/0.2) | 290 | **20.0°** | 0 | 4481 | 18.0 | 0.791 | 99.8 | 249 | 0.033 |

**观察**：
- wing 全部在 20.0°（搜索下界，由用户根据实际 F1 工程经验设定）
- drag 全部在 18.0N（训练数据下界）
- stability 全部 ~99.8（模型输出近乎饱和）
- 场景间唯一有明显区分的是复合 fitness（来自不同权重对 efficiency/power 的重新加权）

### 2.2 为什么 3 个指标没有产生张力

```
给定: speed 固定（如 345 km/h），wing 和 drag 是自由变量

目标1 (stability) : 在模型预测中 ≈ 100，与 wing 和 drag 无关
目标2 (efficiency): downforce/drag → drag↓ 必然增加 → 推 drag 到下界
目标3 (power)     : drag × speed  → drag↓ 必然降低 → 推 drag 到下界

结论: 3 个目标在 drag 维度完全共线。PSO 的最优策略 = drag→18 + wing→20
```

### 2.3 景观分析结论

随机采样 30,000 点的 PCA 投影显示：
- **单目标着色**（stability only）: 高分点 (黄色) 集中为**单一巨大区域**
- **多目标着色**: 由于 efficiency 和 power 的重新加权，黄色区域的形状略有变形，但仍为**单峰结构**
- t-SNE 同样确认无多峰性

**这意味着**：当前代理模型形成的优化景观缺乏多模结构，PSO 的价值主要体现在处理离散变量 (drs=0/1) 和软约束边界的探索，而非在多个局部最优之间跳跃。

---

## 3. 为什么简单多目标不够 —— 更深层的解释

v2 的 3 个目标本质上测量的是**同一维度的不同侧面**：

```
stability ↑  ← (downforce) →  efficiency ↑
                             →  power ↓ (drag×speed)
            ←─────────────────────────────→
                   都在奖励低 drag
```

真实 F1 设计中翼角的 tradeoff 应该是：

```
直线速度:   low wing  →  low drag  →  high top speed  ✅
弯道抓地力: high wing →  high downforce →  high cornering  ✅
                               ↘  high drag →  low top speed  ❌ (直接冲突)
```

这个冲突在当前模型中被抹平了，因为**模型不编码 "高翼角 → 高 downforce" 的物理因果链**（训练数据中 wing 和 downforce 是独立采样的）。

---

## 4. 建议的解决方案：鲁棒多条件优化

### 4.1 核心思路

将优化从 "单点最佳" 改为 **"同一组参数在多种条件下尽可能高效"**。

**具体形式**：每场景不固定 speed，而是定义 3 个评估速度点。PSO 搜索同一组 (wing, drs, downforce, drag)，但分别在 3 个 speed 下评估该组表现。speed 本身也作为 PSO 的搜索变量之一（在场景速度区间内）。

### 4.2 各场景评估速度点

| 场景 | v_low (弯道) | v_mid (中速) | v_high (直道顶速) | 速度搜索区间 |
|------|:---:|:---:|:---:|------|
| S1 Monza | 200 | 280 | 345 | [200, 345] |
| S2 Monaco | 120 | 200 | 300 | [120, 300] |
| S3 Balanced | 180 | 250 | 320 | [180, 320] |
| S4 Wet | 140 | 210 | 290 | [140, 290] |

### 4.3 适应度函数

```
对一组参数 x = (speed, wing, drs, downforce, drag):

  在 v_low 下:   stability_low  = model(x with speed=v_low)
  在 v_mid 下:   stability_mid  = model(x with speed=v_mid)
  在 v_high 下:  stability_high = model(x with speed=v_high)

  F_robust(x) = min(stability_low, stability_mid, stability_high)
              - λ_eff × std(efficiency_low, efficiency_mid, efficiency_high)
              - λ_risk × σ(x)

  其中: efficiency_i = downforce / (drag + ε)
       σ(x) = DeepEnsemble 预测不确定度
```

### 4.4 为什么这能创建张力

| 机制 | 解释 |
|------|------|
| `min()` 操作 | 即使模型在 3 个速度下都预测 ~99.8，不同速度下的输入组合差异（主要是 downforce/drag 在速度变化时的交互）会产生**微小但非零**的稳定性预测差异。`min()` 会放大这种差异，迫使 PSO 找到**跨速度稳健**的配置 |
| `efficiency_spread` | 在高速下 efficiency 高的配置（低 drag 大翼角组合）可能在中低速下效率骤降。惩罚 spread 迫使 PSO 权衡：全速域效率 vs 峰值效率 |
| speed 变为自由变量 | PSO 可以同时优化 "这组参数在哪个 speed 下表现最好" 和 "在另外两个 speed 下表现如何"——创造了额外的搜索空间维度 |

### 4.5 预期结果

| 场景 | wing 预期 | 理由 |
|------|:---:|------|
| S1 Monza | 22-25° | 高速下追求稳定性（low/mid speed 的 min 约束），但不能太高（high speed 的 efficiency 会崩） |
| S2 Monaco | 28-33° | 低速下追求效率（downforce 在低速时更有用），高速下可以接受稍低 efficiency |
| S3 Balanced | 24-28° | 三点均衡 |
| S4 Wet | 30-35° | 稳定性优先，容忍 wide spread |

### 4.6 与 v2 的关键区别

| 维度 | v2 (多目标加权) | v3 (鲁棒多条件) |
|------|------|------|
| speed | 固定为场景顶速 | **自由变量**，在 3 个评估点间变化 |
| 目标数量 | 3 个（stability, efficiency, power） | 3×3 = 9 个（3 speed × 3 指标）→ 压缩为 2 个（min_stability, spread_efficiency） |
| tension 来源 | 权重之间（语义上） | **速度变化**引起的性能差异（操作上） |
| PSO 搜索空间 | 4D (wing, drs, df, drag) | **5D** (speed, wing, drs, df, drag) |
| 模型需求 | 需要模型对 wing 敏感 | **不要求**模型对 wing 敏感——只要求模型对不同 speed 输入产生不同输出（已经满足） |

---

## 5. 需修改的文件

| 文件 | 改动 | 估时 |
|------|------|:---:|
| `src/scenarios/scenario_def.py` | 每场景新增 `eval_speeds` 列表，speed 恢复为搜索区间（非固定值） | 10min |
| `src/optimization/fitness.py` | `FitnessMultiObjective` 重写为多评估点版本 | 20min |
| `src/scenarios/scenario_runner.py` | 适配新 fitness 签名 | 10min |
| `src/visualization/plot_scenarios.py` | 已有函数无需修改（Pareto/景观图适配新的 component 输出） | 5min |
| `experiments/run_multi_scenario.py` | 无需结构性修改 | — |
| **合计** | | **~45min** |

---

## 6. 风险评估

| 风险 | 等级 | 应对 |
|------|:---:|------|
| 模型在 3 个速度下的输出依然无差异 → `min()` 退化 | 中 | 引入 `efficiency_spread` 作为独立张力源——它计算自参数本身，不依赖模型输出 |
| speed 变成自由变量后搜索收敛变慢 | 低 | speed 区间缩小（3 个评估点定义的 feasible range 远小于原始 [80,360]） |
| 景观分析仍然显示单峰 | 低 | 即使单峰，鲁棒优化提供了 **v2 没有的新叙事**："PSO 在多种工况下搜索稳健解" |

---

## 7. 评估要点

请在审阅时关注以下问题：

1. **方案 4.2 的评估速度点**：各场景的 v_low/v_mid/v_high 取值是否合理？是否与真实 F1 赛道的速度剖面匹配？
2. **方案 4.3 的 F_robust 公式**：`min(stability)` + `efficiency_spread` 的组合是否充分？是否需要额外的约束（如 downforce 下限）？
3. **v1/v2/v3 三个版本的关系**：是否需要保留 v2 的多目标加权作为对照组？还是在 v3 之后废弃 v2？
