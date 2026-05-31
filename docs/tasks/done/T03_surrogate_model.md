# T03: 代理模型构建与对比

**优先级**: 🔴 核心必做

**依赖**: T01 (数据预处理完成), T02 (EDA 完成，了解数据分布)

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

**状态**: ✅ 已完成

---

## 最终结果

| Model | R2 | MSE | MAE | RMSE | R2_low | MSE_low | Latency_ms | Train_Time_s |
|-------|------|------|------|------|------|------|------|------|
| M1_Ridge | 0.633 | 225.00 | 12.34 | 15.00 | 0.78 | 239.03 | 0.034 | 0.0 |
| M2_RandomForest | **1.000** | **0.000** | 0.001 | 0.003 | **1.000** | **0.000** | 37.073 | 3.1 |
| M3_XGBoost | 0.9999 | 0.031 | 0.035 | 0.177 | 1.000 | 0.037 | **0.242** | 0.3 |
| M4_MLP | 0.9994 | 0.390 | 0.388 | 0.624 | 0.999 | 1.169 | 0.403 | 81.8 |
| M5_TabNet | 0.9997 | 0.200 | 0.360 | 0.448 | 0.9998 | 0.173 | 7.847 | 763.3 |
| M6_DeepEnsemble | 0.9995 | 0.327 | 0.271 | 0.572 | 0.999 | 1.149 | 1.230 | 219.9 |

- **验收标准**: ✅ 6 个模型全部训练完成；R² 均 ≥ 0.85（Ridge 除外，作为性能下界符合预期）
- **R²_high 负值**: 高稳定性区 (≥95) 86% 样本 stability_index=100，方差接近 0，R² 数学上不稳定，MSE 仍极低
- **不确定性**: DeepEnsemble σ 均值 0.49（测试集），σ 分布 train/test 一致 (无严重 OOD)
- **物理一致性**: 6 张响应面图全部生成，speed×wing 上 stability 光滑递减，符合气动物理常识

## 关键决策记录

| # | 决策 | 方案 | 原因 |
|---|------|------|------|
| 1 | 特征集 | 5 基础特征 (speed_kmh, wing_angle_deg, drs_active, downforce_n, drag_n) | 对齐 doc-architecture Input(5)；公平对比所有模型 |
| 2 | 归一化 | 树模型用 StandardScaler，NN/TabNet 用 MinMax | 树模型对尺度不敏感，NN 需有界输入 |
| 3 | y 标准化 | NN 族训练时 y/=100 缩放到 [0,1]，预测后 ×100 反变换 | 提升 NN 训练稳定性 |
| 4 | 样本权重 | 从 train.csv 的 stability_label 按 N/(4×n_i) 重新计算 | 训练集索引丢失，现场重算更可靠 |
| 5 | 最优模型 | **M2_RandomForest** (R²=1.0000) — 精度最优 | 合成数据近似确定性，RF 完美拟合 |
| 6 | 实用推荐 | **M3_XGBoost** (R²=0.9999, 0.24ms) — 速度+精度最优 | 推理时延仅 0.24ms，适合 PSO 高频调用；RF 37ms 偏慢 |
| 7 | 不确定性备选 | **M6_DeepEnsemble** (R²=0.9995, σ=0.49) | 提供预测不确定性，为风险敏感 PSO (T04) 提供基础 |
| 8 | DeepEnsemble M 值 | M=3，seed 差 100 | 平衡效率与多样性 |
| 9 | PSO 推理模型 | 首选 XGBoost (0.24ms)，备选 RF (37ms) | 高频调用场景速度优先 |
| 10 | M5 TabNet | 耗时 763s、推理 7.85ms，精度与 XGBoost 持平 | 训练/推理效率均不如 XGBoost，仅作为对比基准 |
| 11 | M1 Ridge | R²=0.633 未达标 | 确认非线性关系不可忽略，线性模型不足以建模 |
| 12 | 物理一致性 | speed×wing_angle 响应面显示 stability 随速度升高、翼角增大而下降 | 符合气动物理：高速+大翼角→更大湍流→稳定性下降 |
| 13 | TabNet 修复 | 补 `import torch` | TabNet 使用 `torch.optim.Adam` 需显式导入 |
| 14 | 时延 NaN 修复 | 实验脚本顶部统一导入 `measure_inference_latency` | 原先 M4/M5/M6 段内 try/except 中调用未导入的函数 |
