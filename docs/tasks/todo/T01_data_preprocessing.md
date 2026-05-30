# T01: 数据预处理与特征工程

**优先级**: 🔴 核心必做

**依赖**: 无

**负责模块**: `src/preprocessing/`

---

## 任务描述

对 Kaggle 原始数据集（~150k 样本）进行清洗、归一化、划分和特征工程，为后续建模提供高质量输入。同步处理数据不平衡问题——高稳定性样本（≥90）占优导致的偏态。

---

## 分步执行

### 1.1 数据清洗
- [ ] 缺失值检测：若有缺失，按列统计缺失率；<5% 删除样本，5-20% 填充（中位数/众数），>20% 视情况删除该列
- [ ] 重复样本检测与去重
- [ ] 异常值检测：任选 IQR / Z-Score / Isolation Forest 之一实施
- [ ] 数据类型统一（确保所有列为 float32）

**输出**: `data/processed/clean_dataset.csv`

### 1.2 数据归一化
- [ ] 实现 Min-Max 归一化（推荐用于神经网络）
- [ ] 实现 StandardScaler（推荐用于树模型）
- [ ] 保存两种归一化结果，模型阶段对比选择

**输出**: `data/processed/X_train_mm.csv` / `X_train_ss.csv` 等

### 1.3 数据集划分
- [ ] 70% / 15% / 15% 三层划分
- [ ] 固定 `random_state=42`
- [ ] 对 stability_index 进行分桶（如 <60, 60-90, >90），使用分层采样保证各子集分布一致

### 1.4 特征工程
必做:
- [ ] Pearson/Spearman 相关性矩阵计算
- [ ] stability_index 三档离散标记 (low/mid/high)

推荐:
- [ ] 物理复合特征: `force_ratio = downforce_n / drag_n`, `speed_wing = speed_kmh * wing_angle_deg`
- [ ] 对关键连续变量做分位数离散化分桶 (作为分类特征供树模型使用)

可选:
- [ ] 多项式特征扩展 (二次交互项)

### 1.5 数据不平衡标记
- [ ] 统计各档位样本占比，输出 `reports/imbalance_report.md`
- [ ] 计算低分样本的 sample_weight（供后续训练加权使用）

---

## 验收标准

- [ ] 清洗后数据集无缺失值、无重复、无极端异常
- [ ] 训练/验证/测试集分布在各 stability_index 档位上偏差 < 3%
- [ ] 至少完成 Min-Max 归一化
- [ ] 导出 `imbalance_report.md` 明确标注偏态程度

## 输入
- `data/raw/actaruslab_f1_telemetry_2026.csv` (或从 Kaggle 重新下载)

## 输出
```
data/processed/
├── clean_dataset.csv
├── train.csv / valid.csv / test.csv
├── X_train_mm.npy, y_train.npy   (或 csv)
└── scaler.pkl
reports/imbalance_report.md
```

## 弹性空间

> - 复合特征是否有效需后续消融实验验证；若无增益可舍弃
> - 两次归一化对比可在 T03 模型阶段完成；时间紧迫则仅保留 Min-Max
> - 异常值检测方法可任选其一，无需实现多种

## 关键决策记录

> (执行过程中记录：选择了何种缺失值策略、为何选择某归一化方案、复合特征效果如何等)
