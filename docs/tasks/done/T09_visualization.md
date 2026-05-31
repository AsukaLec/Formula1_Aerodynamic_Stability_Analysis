# T09: 结果可视化汇总

**优先级**: 🔴 核心必做

**依赖**: T02 (EDA), T03 (模型对比), T04 (PSO 收敛), T06 (场景分析), T07 (风险热力图, ✅)

**负责模块**: `src/visualization/`

---

## 任务描述

汇总所有实验阶段产生的可视化需求，统一生成高质量图表。确保每张图表有清晰的标题、轴标签、图例，适合直接嵌入结题报告。

---

## 分步执行

### 9.1 必须图表清单

| 序号 | 图表 | 数据来源 | 说明 |
|------|------|----------|------|
| F1 | 数据分布图 | T02 | stability_index 直方图 + 偏态标注 |
| F2 | 相关性热力图 | T02 | 全变量相关矩阵 |
| F3 | 模型对比柱状图 | T03 | 各模型 R² / MSE / 推理速度 |
| F4 | 训练损失曲线 | T03 | 各 NN 模型训练 loss 下降 |
| F5 | PSO 收敛曲线 | T08 | 各实验组迭代 fitness 变化 |
| F6 | 最优解箱线图 | T08 | 多组实验最优 fitness 分布 |
| F7 | 场景雷达图 | T06 | 各场景最优解特征对比 |
| F8 | SHAP Summary | T06 | 全局特征重要性 |
| F9 | SHAP Waterfall ×2 | T06 | S1 和 S2 最优解决策归因 |
| F10 | 约束灵敏度图 | T06 | 龙卷风图 |

### 9.2 推荐图表清单

| F11 | Porpoising 风险热力图 | T07 | DRS=0 / DRS=1 |
| F12 | 响应面 3D 图 | T03 | speed × wing → stability |
| F13 | 消融实验对比 | T08 | 移除各模块的性能降幅 |

### 9.3 通用规范

- [x] 尺寸: ≥ 1920×1080 或 8×6 英寸
- [x] DPI: ≥ 150 (适合报告嵌入)
- [x] 中英文混排: 标题/轴标签统一用英文
- [x] 配色: 使用 seaborn colorblind palette，色盲友好
- [x] 导出格式: PNG (主, 15张全部导出)

---

## 验收标准

- [x] 生成 F1~F10 全部图表 (F11~F13 为推荐)
- [x] 所有图表统一风格、清晰可读
- [x] 图表文件位于 `figures/` 根目录下

## 输入
- `figures/eda/`, `figures/models/`, `figures/pso/`, `figures/scenarios/`, `figures/porpoising_risk/`

## 输出
```
figures/
├── 01_distribution.png       (F1)
├── 02_correlation.png        (F2)
├── 03_model_comparison.png   (F3)
├── 04_loss_curves.png        (F4)
├── 05_pso_convergence.png    (F5)
├── 06_best_fitness_box.png   (F6)
├── 07_scenario_radar.png     (F7)
├── 08_shap_summary.png       (F8)
├── 09_shap_waterfall_S1.png  (F9)
├── 10_shap_waterfall_S2.png  (F9)
├── 11_sensitivity_tornado.png(F10)
├── 12_risk_heatmap_drs0.png  (F11, 推荐)
├── 13_risk_heatmap_drs1.png  (F11, 推荐)
├── 14_response_surface.png   (F12, 推荐)
└── 15_ablation_study.png     (F13, 推荐)
```

## 弹性空间

> - 推荐图表视前置模块完成情况决定是否生成
> - 若时间紧迫，可简化图表风格 (如去掉 3D 响应面的交互性)
> - 图表编号和命名可根据实际生成情况调整

---

## 执行记录

**执行日期**: 2026-05-31

**编排脚本**: `experiments/run_visualization.py`

**生成策略**:
| 图表 | 生成方式 | 说明 |
|------|----------|------|
| F1, F2, F3 | 从 scratch 重新生成 | F1/F2 从 processed data 直接绘制；F3 从 `model_comparison.csv` 合并 R²/MSE/Latency 三图为一张 |
| F4, F5, F6, F13 | 复制已有 | 从 `figures/models/`、`figures/pso/` 复制并重命名 — PSO 实验耗时过长 |
| F7, F10 | 从缓存重新生成 | F7 从 `outputs/scenarios/*/stats.json` 加载数据重绘雷达图；F10 合并 S1+S2 龙卷风图 |
| F8, F9 | 复制已有 | 从 `figures/scenarios/` 复制 — SHAP 需要加载模型并运行 explainer，复制已有高质量图更高效 |
| F11, F12 | 复制已有 | 从 `figures/porpoising_risk/`、`figures/models/` 复制 — 需 DeepEnsemble autograd / KNN 响应面计算 |

**关键决策**:
- F5 取 T08 版 `convergence_comparison.png`（含 Baseline+Exp1-4 多曲线），不使用 T04 版
- F10 将 S1/S2 两个独立 tornado 合并为一张 1×2 并排图
- 统一风格：DPI=150, seaborn colorblind palette, English 标签, DejaVu Sans 字体
- 15 张图全部成功生成，F1-F13 全覆盖（核心+推荐）
