# T01: 数据预处理与特征工程

**优先级**: 🔴 核心必做

**依赖**: 无

**负责模块**: `src/preprocessing/`

---

## 任务描述

对 Kaggle 原始数据集（~150k 样本）进行清洗、归一化、划分和特征工程，为后续建模提供高质量输入。同步处理数据不平衡问题——高稳定性样本（≥95）占优导致的偏态。

---

## 分步执行

### 1.1 数据清洗
- [x] 缺失值检测：若有缺失，按列统计缺失率；<5% 删除样本，5-20% 填充（中位数/众数），>20% 视情况删除该列
- [x] 重复样本检测与去重
- [x] 异常值检测：任选 IQR / Z-Score / Isolation Forest 之一实施
- [x] 数据类型统一（确保所有列为 float32）

**输出**: `data/processed/clean_dataset.csv`

### 1.2 数据归一化
- [x] 实现 Min-Max 归一化（推荐用于神经网络）
- [x] 实现 StandardScaler（推荐用于树模型）
- [x] 保存两种归一化结果，模型阶段对比选择

**输出**: `data/processed/X_train_mm.csv` / `X_train_ss.csv` 等

### 1.3 数据集划分
- [x] 70% / 15% / 15% 三层划分
- [x] 固定 `random_state=42`
- [x] 对 stability_index 进行 4 档分桶（<30 severe / 30-60 moderate / 60-95 mild / ≥95 stable），使用分层采样保证各子集分布一致

### 1.4 特征工程
必做:
- [x] Pearson/Spearman 相关性矩阵计算
- [x] stability_index 四档离散标记 (severe/moderate/mild/stable)

推荐:
- [x] 物理复合特征: `force_ratio = downforce_n / drag_n`, `speed_wing = speed_kmh * wing_angle_deg`
- [x] 对关键连续变量做分位数离散化分桶 (作为分类特征供树模型使用)

可选:
- [ ] 多项式特征扩展 (二次交互项)

### 1.5 数据不平衡标记
- [x] 统计各档位样本占比，输出 `reports/imbalance_report.md`
- [x] 计算低分样本的 sample_weight（供后续训练加权使用）

---

## 验收标准

- [x] 清洗后数据集无缺失值、无重复、无极端异常
- [x] 训练/验证/测试集分布在各 stability_index 档位上偏差 < 3%
- [x] 至少完成 Min-Max 归一化
- [x] 导出 `imbalance_report.md` 明确标注偏态程度

## 输入
- `data/raw/actaruslab_f1_telemetry_2026.csv` (或从 Kaggle 重新下载)

## 输出
```
data/processed/
├── clean_dataset.csv
├── train.csv / valid.csv / test.csv
├── X_train_mm.npy, X_valid_mm.npy, X_test_mm.npy
├── X_train_ss.npy, X_valid_ss.npy, X_test_ss.npy
├── y_train.npy, y_valid.npy, y_test.npy
├── scaler_mm.pkl, scaler_ss.pkl
├── sample_weights.npy
└── features_dataset.csv
reports/imbalance_report.md
```

额外产出:
- `porpoising_flag` 列（stability_index < 95 = 1）已写入 `features_dataset.csv`，供 T07 直接使用

## 弹性空间

> - 复合特征是否有效需后续消融实验验证；若无增益可舍弃
> - 两次归一化对比可在 T03 模型阶段完成；时间紧迫则仅保留 Min-Max
> - 异常值检测方法可任选其一，无需实现多种

## 关键决策记录

| 决策 | 方案 | 原因 |
|------|------|------|
| Python 环境 | `/mnt/e/python313/python.exe` (Windows 主机) | WSL 系统 Python 受 PEP 668 保护，主机已预装全部依赖 |
| 异常值检测 | IQR, factor=3.0, 仅作用于输入特征 | stability_index 为目标变量不应过滤 |
| 归一化方案 | Min-Max + StandardScaler 均保存 | T03 模型阶段对比选择 |
| 数据分桶 | stability_index: <30 (severe), 30-60 (moderate), 60-95 (mild), ≥95 (stable) | 4 档对齐物理阈值: 95 为海豚跳分界线, 30/60 细分严重程度 |
| 附加特征 | porpoising_flag = (stability_index < 95) | 供 T07 风险热力图使用 |
| 数据集划分 | 70/15/15 分层采样 (stratify on stability_label) | 确保各子集分布一致 |
| 划分工具 | sklearn train_test_split (2次调用) | 标准做法 |
| 复合特征 | force_ratio, speed_wing | 物理意义明确 (CL/CD 比)，后续消融验证 |
| 样本权重 | 类别平衡权重: weight_i = N / (K * n_i) | 缓解高稳定性样本偏态 |
| 缺失值 | 无缺失值 | 数据集质量高 |
| 数据集格式 | .csv (可读) + .npy (高效) 双格式 | 兼顾可读性与加载速度 |
