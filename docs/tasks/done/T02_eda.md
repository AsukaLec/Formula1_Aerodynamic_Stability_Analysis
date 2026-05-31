# T02: 探索性数据分析 (EDA)

**优先级**: 🔴 核心必做

**依赖**: T01 (数据预处理完成)

**负责模块**: `src/visualization/plot_eda.py` + `notebooks/01_eda.ipynb`

**状态**: ✅ 已完成

---

## 任务描述

对预处理后的数据集进行全面的统计分析和可视化探索，揭示数据分布规律、变量间关系及数据偏态特征，为模型选择和不平衡处理策略提供依据。

---

## 分步执行

### 2.1 基础统计
- [x] 各变量: 均值、标准差、min、max、Q1、Q2、Q3
- [x] 偏度 (Skewness) 与峰度 (Kurtosis)
- [x] 输出 `reports/statistics_report.md`

### 2.2 分布可视化

必做:
- [x] stability_index 直方图 + KDE (重点突出偏态)
- [x] 各输入变量的直方图/KDE

推荐:
- [x] 按 DRS 状态 (0/1) 分组，绘制 stability_index 箱线图/小提琴图对比
- [x] 按速度区间 (低速 [80-180] / 中速 [180-280] / 高速 [280-360]) 分组分析稳定性分布

### 2.3 相关性分析
- [x] 全变量 Pearson 相关系数矩阵 + 热力图
- [x] (可选) Spearman 秩相关系数（对非线性关系更敏感）

### 2.4 变量间关系
- [x] Pairplot 全局概览（5x5 连续变量，采样 8k 点）
- [x] 重点散点图: `speed_kmh vs stability_index` 按 DRS 颜色分组
- [x] `wing_angle_deg vs stability_index` 散点图
- [x] `speed_kmh vs wing_angle_deg` 联合分布密度热力图 (hexbin)

### 2.5 数据不平衡量化
- [x] 绘制 stability_index 的累计分布函数 (CDF)
- [x] 计算 P(stability<95) 与 P(stability≥95) 比率（海豚跳比例）
- [x] 按 4 档 (severe<30 / moderate 30-60 / mild 60-95 / stable≥95) 统计各档稀缺程度

---

## 验收标准

- [x] `statistics_report.md` 包含完整统计量
- [x] 生成的图表 12 张（超过 6 张验收标准）
- [x] 明确量化数据不平衡程度：stable/unstable = 6.2:1, unstable = 13.98%
- [x] 发现 2 个分布特征：stability_index 严重左偏 + 上限截断 (86% 在 100)；downforce_n/drag_n 正偏态

## 输入
- `data/processed/train.csv`

## 实际产出
```
figures/eda/
├── README.md                          # 图表数值说明与解读指南 (自动生成)
├── distribution_stability.png
├── distribution_features.png
├── stability_by_drs_boxplot.png
├── stability_by_speed_group.png
├── correlation_pearson_heatmap.png
├── correlation_spearman_heatmap.png
├── pairplot.png
├── speed_vs_stability_scatter.png
├── wing_vs_stability_scatter.png
├── speed_wing_stability_hexbin.png    # hexbin: 颜色 = mean(stability_index)
├── speed_wing_stability_hist2d.png    # hist2d: 颜色 = mean(stability_index)
└── imbalance_cdf.png
reports/statistics_report.md
```

额外产出：
- Spearman 热力图（已生成，发现 downforce/drag Spearman=0.98，确认单调非线性共线）
- `figures/eda/README.md` — `run_eda()` 自动生成，逐张图说明含义/轴/数值/解读

---

## 弹性空间

> - Pairplot 采样 8000 点，渲染正常，无性能问题
> - 交互式可视化 (Plotly) 跳过（时间优先级）
> - Pearson 热力图显示 downforce_n~drag_n=0.94, speed~downforce=0.82, speed~drag=0.78；已在报告中标注共线性风险

## 关键决策记录

| # | 决策 | 方案 | 原因 |
|---|------|------|------|
| 1 | 数据源 | `train.csv` (104,999 样本) | T01 分层采样，分布与全量一致 |
| 2 | 速度分组 | [0,180,280,400] 三档 | 对齐 F1 低速弯/中速弯/高速直道物理场景 |
| 3 | seaborn API | 使用 `hue` + `legend=False` 模式 | 避免 v0.14 `palette` 弃用警告 |
| 4 | DRS 分组图 | str 类型转换 `drs_active` | seaborn palette dict 需要 string keys |
| 5 | 速度组 KDE | `warn_singular=False` | Low/Mid 速度组 stability_index 方差为 0 (全部 100) |
| 6 | Pairplot | 排除 `drs_active` 二值变量，5×5 连续变量，采样 8k | 二值变量不对角线 KDE 无意义，8k 采样保证渲染速度 |
| 7 | 相关性方法 | Pearson + Spearman 双热力图 | Spearman 暴露 downforce/drag 单调关系 (0.98)，比 Pearson 更敏感 |
| 8 | 联合分布图 | 从密度热力图 → 稳定性热力图 (颜色 = mean stability) | speed/wing 独立且准均匀，计数热力图无信息；稳定性热力图直接服务优化 |
| 9 | 联合图 bin 数 | bins=120, gridsize=120 | 60 格分辨率偏低，120 格细节足够且不显噪声 |
| 10 | 联合图 dpi | 300（覆盖全局 150） | 高分辨率便于观察稳定区域边界 |
| 11 | 联合图双版本 | hexbin + hist2d 各一张 | 六边形平滑各向同性好，矩形网格便于精确定位 |
| 12 | Figure Guide | `generate_figure_guide()` 写 `figures/eda/README.md` | 自动生成，图与说明永不脱节；`run_eda()` 末尾调用 |
| 13 | 密度热力图删除 | 删除旧 `speed_wing_joint_*.png` | 无信息量（均匀矩形），被稳定性热力图取代 |

### 关键分布发现

1. **stability_index 严重左偏 (skew=-2.90, kurtosis=7.07)**: 86% 样本集中在 100（上限截断）。低分区域呈现均匀-稀疏分布，严重样本比 moderate 更多 (6.38% vs 3.01%)。
2. **共线性风险**: downforce_n 与 drag_n Pearson=0.94, speed 与 downforce=0.82, speed 与 drag=0.78。建模时可能需要降维或正则化处理。
3. **DRS 对稳定性影响有限**: 箱线图显示 DRS ON/OFF 的稳定性中位数均接近 100，但 DRS ON 时下尾分布略有差异。
4. **低速组稳定性方差为零**: 低速组内所有样本 stability_index=100，无有效变异。
