# T07: Porpoising 风险可解释性

**优先级**: 🟡 推荐 (直接对应开题报告"海豚跳"痛点)

**依赖**: T03 (代理模型就绪)

**负责模块**: `src/visualization/plot_porpoising.py`

---

## 任务描述

利用代理模型的局部梯度信息，识别参数空间中气动稳定性急剧变化的区域（即"海豚跳"高风险区），生成风险热力图并叠加 PSO 最优解分析其安全性。

---

## 分步执行

### 7.1 风险量化

定义:
```
Risk(x) = ||∇f(x)|| ≈ |∂f/∂v| + |∂f/∂α|
```

- [x] 在 `speed_kmh × wing_angle_deg` 二维网格上计算代理模型输出的梯度
- [x] 若模型为 NN: PyTorch autograd 直接计算梯度 (DeepEnsemble refined, 3 MLP members)
- [x] 若模型为 XGBoost: 使用局部扰动 + 数值差分近似 (已实现备选方案 compute_xgboost_risk)

### 7.2 热力图生成

- [x] 分别对 DRS=0 和 DRS=1 两种状态生成风险热力图
- [x] 颜色深浅表示海豚跳风险高低 (log-scale + percentile normalisation for visibility)
- [x] 叠加 T06 中各场景 PSO 最优解的位置标注
- [x] 分析: 最优解是否在低风险区？是否靠近高风险边界？

### 7.3 物理直觉验证

- [x] 标出高速 + 小翼角区域 (传统认知中的易失稳区) — Zone A 矩形标注
- [x] 标出低速 + 大翼角区域 (传统认知中的稳定区) — Zone B 矩形标注
- [x] 对照领域知识，检查风险热力图是否符合物理预期
- [x] 若不符合 → 可能模型过拟合或数据偏差，需记录分析 (已记录: 见关键决策)

---

## 验收标准

- [x] 生成 DRS=0 和 DRS=1 两组 Porpoising 风险热力图
- [x] 图上叠加了 T06 场景最优解的位置 (with Zone A/B physics annotations)
- [x] 报告中有一段关于最优解安全性的分析 (console output + 关键决策记录)
- [ ] (推荐) 风险分布与气动物理直觉一致 → **未通过**: 模型平滑插值导致梯度无法表征物理风险, 已记录分析

## 输出
```
figures/porpoising_risk/
├── risk_heatmap_drs0.png          (log-scale, Zone A/B annotated)
├── risk_heatmap_drs1.png          (log-scale, Zone A/B annotated)
├── risk_vs_optimal_overlay.png    (side-by-side comparison)
├── stability_surface_drs0.png     (predicted stability_index, DRS=0)
└── stability_surface_drs1.png     (predicted stability_index, DRS=1)
```

## 输入
- `outputs/models/deep_ensemble_refined/mlp_{0,1,2}.pt` (实际使用, 非 best_model.pt)
- `outputs/scenarios/*/best_solution.json`
- `data/processed/X_train_mm.npy` + `scaler_mm.pkl`

## 弹性空间

> - 此模块不依赖 PSO，可与 T06 并行开发
> - 若梯度的解析计算困难，数值差分近似完全可以接受
> - 若此模块被跳过，需在报告中说明"海豚跳风险分析作为未来工作"
> - 可扩展: 不仅分析梯度范数，还分析 Hessian 特征值（曲率分析）

## 关键决策记录

### 1. 模型选择与梯度计算方法
- **选用模型**: DeepEnsemble refined (3 × MLP, `outputs/models/deep_ensemble_refined/`)
- **梯度计算**: PyTorch autograd, 逐成员计算后取平均梯度
- **网格分辨率**: 80×80, KNN(K=10) 估算 downforce_n 和 drag_n
- **备选方案**: XGBoost + 中心差分 (已实现在 `compute_xgboost_risk()`, 未启用)

### 2. 风险分布特征
- **核心发现**: 风险分布极度右偏 (median ≈ 0.001, max ≈ 2.5) — 模型预测面在绝大多数区域非常平坦
- **原因分析**:
  1. 训练数据中高稳定性样本占绝对多数, MLP 在大部分区域学到近乎恒定的高稳定性预测
  2. 梯度范数反映的是"模型插值的局部变化率"而非"物理风险"
- **改进方向**: 使用 DeepEnsemble 的 σ(x) (预测不确定性) 作为补充风险指标

### 3. 物理直觉验证结果 (v2 — 扩展网格后)
- **v1 (训练数据范围网格 [100,365]×[5,35])**: 高速+小翼角 < 低速+大翼角 — 与直觉相反
- **v2 (PSO bounds 网格 [80,360]×[0,45])**: 
  - Zone A (高速+小翼角) 平均风险 0.0013 > Zone B (低速+大翼角) 平均风险 0.0002 — **符合气动物理直觉**
  - DRS ON/OFF 风险比 1.01 — DRS ON 略微增加风险, 符合物理预期
- **根因**: v1 网格截断了 wing=[0,5°) 的真实风险上升区域; 扩展网格后模型在 OOD 边缘的梯度回升
- **结论**: 网格边界设置对 gradient-based risk 结果有显著影响, 应使用 PSO bounds 以确保覆盖所有可能解

### 4. 场景最优解安全性 (v2)
- S1 Monza (v=280, α=0): p74.5, **[OOD-wing]** (α=0 超出训练范围 [5,35])
- S2 Monaco (v=158, α=20): p36.0, 参数均在训练分布内
- S3 Balanced (v=200, α=10): p55.1
- S4 Wet (v=155, α=15): p41.7
- **总体**: OOD 解的 risk 被合理标记; 所有解风险值仍较低, 但扩展网格后区分度提升

### 5. 可视化改进
- 使用 log-scale color mapping (LogNorm) 替代线性色阶以增强对比度
- 网格扩展至 PSO_PARAM_BOUNDS (speed [80,360], wing [0,45])
- 添加 p90 风险边界等高线
- **OOD 区域灰色斜线遮罩** + "hatch = OOD extrapolation" 标注
- **GridSpec 布局修复**: overlay 图使用 `add_gridspec(1, 3)` + 独立 cax 轴, 避免 shared colorbar 挤压右子图
- 稳定性预测面图 (`stability_surface_drs*.png`) 同步扩展网格 + OOD 标注
- 用矩形框标注 Zone A (高速+小翼角) / Zone B (低速+大翼角) 物理直觉区域
- 点标记增加深色边框增强可见性

### 6. 布局修复 (Bugfix)
- **问题**: `fig.colorbar(c, ax=[ax0,ax1])` 使 shared colorbar 在 tight_layout 下挤压右子图
- **解决方案**: 改用 `matplotlib.gridspec.GridSpec` 三列布局 `[1, 1, 0.04]`, colorbar 独占 cax
