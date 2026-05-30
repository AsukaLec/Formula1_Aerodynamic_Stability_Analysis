# T02: 探索性数据分析 (EDA)

**优先级**: 🔴 核心必做

**依赖**: T01 (数据预处理完成)

**负责模块**: `src/visualization/plot_eda.py` + `notebooks/01_eda.ipynb`

---

## 任务描述

对预处理后的数据集进行全面的统计分析和可视化探索，揭示数据分布规律、变量间关系及数据偏态特征，为模型选择和不平衡处理策略提供依据。

---

## 分步执行

### 2.1 基础统计
- [ ] 各变量: 均值、标准差、min、max、Q1、Q2、Q3
- [ ] 偏度 (Skewness) 与峰度 (Kurtosis)
- [ ] 输出 `reports/statistics_report.md`

### 2.2 分布可视化

必做:
- [ ] stability_index 直方图 + KDE (重点突出偏态)
- [ ] 各输入变量的直方图/KDE

推荐:
- [ ] 按 DRS 状态 (0/1) 分组，绘制 stability_index 箱线图/小提琴图对比
- [ ] 按速度区间 (低速 [80-180] / 中速 [180-280] / 高速 [280-360]) 分组分析稳定性分布

### 2.3 相关性分析
- [ ] 全变量 Pearson 相关系数矩阵 + 热力图
- [ ] (可选) Spearman 秩相关系数（对非线性关系更敏感）

### 2.4 变量间关系
- [ ] Pairplot 全局概览（6×6 或 5×5 排除 downforce/drag 相关性过高对）
- [ ] 重点散点图: `speed_kmh vs stability_index` 按 DRS 颜色分组
- [ ] `wing_angle_deg vs stability_index` 散点图
- [ ] `speed_kmh vs wing_angle_deg` 联合分布密度热力图

### 2.5 数据不平衡量化
- [ ] 绘制 stability_index 的累计分布函数 (CDF)
- [ ] 计算 P(y>90) 与 P(y<90) 比率
- [ ] 识别低稳定性样本（<60）是否极度稀缺

---

## 验收标准

- [ ] `statistics_report.md` 包含完整统计量
- [ ] 生成的图表 ≥ 6 张（直方图、热力图、箱线图、散点图矩阵等）
- [ ] 明确量化数据不平衡程度（高分/低分样本比）
- [ ] 发现至少 1 个值得在报告中讨论的分布特征

## 输入
- `data/processed/train.csv` (或完整 clean_dataset.csv)

## 输出
```
figures/eda/
├── distribution_stability.png
├── distribution_features.png
├── correlation_heatmap.png
├── stability_by_drs_boxplot.png
├── speed_vs_stability_scatter.png
├── wing_vs_stability_scatter.png
└── speed_wing_joint_heatmap.png
reports/statistics_report.md
```

## 弹性空间

> - Pairplot 若数据量大导致渲染慢，可采样至 5k-10k 点绘制
> - 交互式可视化 (Plotly) 为可选项，根据展示需求决定
> - 若发现 speed 与 downforce/drag 高度相关，可在 EDA 报告中标注共线性风险

## 关键决策记录

> (执行过程中记录发现的关键分布特征、值得关注的异常模式等)
