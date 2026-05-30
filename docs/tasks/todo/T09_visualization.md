# T09: 结果可视化汇总

**优先级**: 🔴 核心必做

**依赖**: T02 (EDA), T03 (模型对比), T04 (PSO 收敛), T06 (场景分析), T07 (风险热力图, 可选)

**预计耗时**: 1-2 天

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

- [ ] 尺寸: ≥ 1920×1080 或 8×6 英寸
- [ ] DPI: ≥ 150 (适合报告嵌入)
- [ ] 中英文混排: 标题/轴标签统一用中文或英文，不混用
- [ ] 配色: 使用 seaborn/matplotlib 默认主题，确保色盲友好
- [ ] 导出格式: PNG (主) + PDF (矢量备份)

---

## 验收标准

- [ ] 生成 F1~F10 全部图表 (F11~F13 为推荐)
- [ ] 所有图表统一风格、清晰可读
- [ ] 图表文件位于 `figures/` 对应子目录下

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
