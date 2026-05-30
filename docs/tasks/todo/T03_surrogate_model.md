# T03: 代理模型构建与对比

**优先级**: 🔴 核心必做

**依赖**: T01 (数据预处理完成), T02 (EDA 完成，了解数据分布)

**预计耗时**: 3-5 天

**负责模块**: `src/models/`

---

## 任务描述

构建多模型对比基线，全面评估从线性回归到深度集成学习在气动稳定性预测任务上的表现，选择最优代理模型。若条件允许，实现 DeepEnsemble 以获取预测不确定性，为后续风险敏感 PSO 提供基础。

---

## 分步执行

### 3.1 候选模型实现

必做 (≥4个):
| 编号 | 模型 | 实现方式 | 备注 |
|------|------|----------|------|
| M1 | Linear Regression | sklearn | 性能下界 |
| M2 | Random Forest | sklearn | 对离散特征友好 |
| M3 | XGBoost | xgboost | 高精度，捕捉阈值效应 |
| M4 | MLP (BP神经网络) | PyTorch | 3-5 隐藏层，ReLU 激活 |

推荐:
| M5 | TabNet | pytorch-tabnet | 表格数据专用 Attention |
| M6 | DeepEnsemble | PyTorch | M 个独立 MLP，M=3~5 |

### 3.2 训练配置

统一标准:
- [ ] 相同训练/验证/测试集 (来自 T01)
- [ ] `random_state=42`
- [ ] 低稳定性样本加权 (sample_weight 来自 T01 的不平衡标记)
- [ ] NN 族模型统一 batch_size、epoch 上限、early stopping patience

### 3.3 评价体系

| 指标 | 最低达标线 | 理想目标 |
|------|------------|----------|
| R² | ≥ 0.85 | ≥ 0.92 |
| MSE | ≤ 0.05 | ≤ 0.02 |
| MAE | ≤ 0.10 | ≤ 0.05 |
| RMSE | ≤ 0.22 | ≤ 0.14 |

额外指标:
- [ ] 推理时延 (ms) — 对 PSO 高频调用至关重要
- [ ] 分区域 R²/MSE (仅低分样本区域、仅高分样本区域)

### 3.4 模型对比与选择

- [ ] 生成 `reports/model_comparison.csv`，包含所有模型在所有指标上的得分
- [ ] 绘制模型对比柱状图
- [ ] 按优先级选择:
  1. 综合性能最优 (R² + 推理速度 加权)
  2. 若 DeepEnsemble 与单一最佳模型精度差 <5%，可选单一模型降低耗时
  3. 若所有 NN 模型 R² < 0.85，启用 XGBoost 后备路径

### 3.5 DeepEnsemble 不确定性估计

> 若 T03.1 中 M6 已实现，则直接用于后续。若未实现且条件允许：

- [ ] 训练 M 个独立 MLP (M=3~5)，仅初始化 seed 不同
- [ ] 对输入 x，计算 μ(x) = mean(f₁(x) … f_M(x))，σ(x) = std(f₁(x) … f_M(x))
- [ ] 可视化: 在训练数据密集区，σ 应小；稀疏区 σ 应大
- [ ] 保存 `models/deep_ensemble/`

### 3.6 物理一致性验证 (推荐)

- [ ] 选 2 个典型工况断面（固定 DRS=0 和 DRS=1）
- [ ] 在 speed × wing_angle 网格上生成 stability 响应面 3D 图
- [ ] 人工检查趋势是否符合气动物理常识
- [ ] 若趋势违背物理常识 → 排查过拟合或数据问题

---

## 验收标准

- [ ] 至少训练并评估 4 个模型
- [ ] 至少 1 个模型 R² ≥ 0.85
- [ ] 生成 `model_comparison.csv` 和对比图
- [ ] 选定最终代理模型并保存其权重文件
- [ ] (推荐) 完成 DeepEnsemble 不确定性估计
- [ ] (推荐) 完成至少 1 组物理一致性响应面验证

## 输入
- `data/processed/train.csv`, `valid.csv`, `test.csv`

## 输出
```
outputs/models/
├── best_model.pt (或 .pkl)
├── deep_ensemble/     (或 mc_dropout/)
│   ├── mlp_0.pt ... mlp_M.pt
├── scaler_X.pkl
└── scaler_y.pkl
reports/model_comparison.csv
figures/models/
├── model_comparison_bar.png
├── training_loss_curves.png
└── response_surface.png
```

## 弹性空间

> - TabNet (M5) 训练调参复杂时可跳过
> - DeepEnsemble (M6) 若训练耗时过长，降级为 MC-Dropout（单模型+推理时多次 forward）
> - 若所有模型 R² < 0.85，需回溯 T01 增强特征工程
> - 物理一致性验证不是硬性要求，但强烈建议完成以增强报告可信度

## 备选路径

> - 主路径: BP神经网络/DeepEnsemble (期望 R² ≥ 0.90)
> - 后备路径: XGBoost (若 NN 路径失败)
> - 兜底路径: Random Forest (若 XGBoost 也不理想)

## 关键决策记录

> (记录模型选择的最终决策及理由、哪些模型被淘汰及原因、DeepEnsemble 的 M 值和 λ 初值等)
