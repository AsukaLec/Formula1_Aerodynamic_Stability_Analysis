# task.md

# 基于代理模型与改进粒子群算法的 F1 赛车气动稳定性优化研究

---

# 1. 项目背景

F1 赛车气动设计高度依赖 CFD 仿真与风洞实验，但传统方法存在：

* 计算成本高
* 迭代速度慢
* 难以覆盖大规模参数空间
* 对瞬态气动现象（如 porpoising / 海豚跳）建模困难

本项目希望通过：

* 数据驱动代理模型（Surrogate Model）
* 粒子群优化（PSO）
* 多场景参数优化
* 风险敏感优化策略

构建一个轻量化、可快速推理的赛车气动稳定性优化框架。

---

# 2. 项目核心目标

本项目包含两个核心任务：

---

## 2.1 任务一：代理模型构建（Regression / Surrogate Modeling）

目标：

建立一个高精度代理模型：

$$
f(x_1,x_2,\dots,x_n)\rightarrow y
$$

其中：

* 输入：

  * 赛车工况参数
  * 气动参数
  * DRS 状态
  * 翼角
  * 速度等

* 输出：

  * Stability Score（0~100）

目标要求：

* 高预测精度
* 高推理速度
* 良好泛化能力
* 能作为 PSO 的适应度函数

---

## 2.2 任务二：参数寻优（Optimization）

目标：

在给定工况条件下：

自动搜索最优气动参数组合。

形式化定义：

$$
x^*=\arg\max f(x)
$$

subject to:

$$
x\in \Omega
$$

其中：

* (f(x)) 为代理模型
* (\Omega) 为物理约束空间

最终实现：

* 不同场景下的最优配置搜索
* 多工况智能调校
* 自动化参数优化

---

# 3. 数据集说明

数据集来源：

Kaggle F1 Aerodynamic Stability Dataset

数据规模：

* 约 150,000 条样本

数据特点：

* 多变量结构化数据
* 非线性关系明显
* 存在高稳定性区域数据占优现象
* 适合：

  * 回归建模
  * 黑盒优化
  * surrogate modeling

---

# 4. 项目总体技术路线

项目总体流程：

```text
数据预处理
    ↓
探索性数据分析（EDA）
    ↓
特征工程
    ↓
代理模型构建
    ↓
模型评估与解释
    ↓
PSO 参数寻优
    ↓
改进 PSO 策略
    ↓
多场景优化实验
    ↓
鲁棒性与泛化验证
    ↓
结果分析与可视化
```

---

# 5. 模块划分

---

# 模块 A：数据预处理

---

## A.1 数据清洗

需要完成：

* 缺失值检查
* 重复值处理
* 异常值检测
* 数据类型统一

输出：

* clean_dataset.csv

---

## A.2 数据归一化

建议：

* StandardScaler
* MinMaxScaler

要求：

比较不同归一化方式对模型效果的影响。

---

## A.3 数据集划分

划分：

* Train
* Validation
* Test

建议：

```python
70% / 15% / 15%
```

要求：

固定 random seed。

---

# 模块 B：探索性数据分析（EDA）

---

## B.1 基础统计分析

包括：

* 均值
* 方差
* 分布
* 偏度
* 峰度

输出：

* statistics_report.md

---

## B.2 可视化分析

必须完成：

### 分布图

* Histogram
* KDE

### 相关性分析

* Correlation Matrix
* Heatmap

### 特征关系分析

* Pairplot
* Scatter Plot

### Stability 分布分析

重点分析：

* 高分区域占优问题
* 数据分布偏斜

---

## B.3 数据分布偏置分析

分析：

$$
P(y<95)\gg P(y≥95)
$$

研究：

* 是否会影响 surrogate model
* 是否会影响 PSO 搜索

输出：

* imbalance_analysis.md

---

# 模块 C：代理模型构建（核心模块）

---

# C.1 Baseline 模型组

禁止只使用 BP 神经网络。

必须建立 baseline。

至少包括：

| 模型                | 类型       |
| ----------------- | -------- |
| Linear Regression | 线性基线     |
| Random Forest     | 集成学习     |
| XGBoost           | Boosting |
| MLP/BP            | 神经网络     |

可选：

* LightGBM
* CatBoost
* TabNet

---

# C.2 模型训练

要求：

* 统一训练集
* 统一评价指标
* 统一随机种子

---

# C.3 评价指标

必须包含：

| 指标   | 用途   |
| ---- | ---- |
| MAE  | 平均误差 |
| MSE  | 均方误差 |
| RMSE | 回归误差 |
| R²   | 拟合优度 |

---

# C.4 模型对比实验

比较：

* 精度
* 泛化能力
* 推理速度
* 稳定性

输出：

* model_comparison.csv
* model_ranking.md

---

# C.5 最终代理模型选择

从 baseline 中选择：

* 精度最高
* 泛化最好
* 推理速度满足 PSO 高频调用需求

的模型作为最终 surrogate model。

---

# 模块 D：模型解释与可信度分析

---

# D.1 残差分析

分析：

* 哪些区域误差最大
* 是否存在系统性偏差

输出：

* residual_analysis.png

---

# D.2 特征重要性分析

要求：

使用：

* SHAP
* Permutation Importance

分析：

* 哪些变量影响最大
* 不同工况下变量贡献变化

---

# D.3 OOD（分布外）风险分析

问题：

PSO 可能搜索到训练数据未覆盖区域。

需要：

* 分析搜索点与训练分布距离
* 防止 surrogate hallucination

可选方案：

* density penalty
* KNN density
* Mahalanobis distance

---

# 模块 E：PSO 参数寻优（核心模块）

---

# E.1 标准 PSO 实现

实现：

* 粒子初始化
* 速度更新
* 位置更新
* 个体最优
* 全局最优

---

# E.2 适应度函数设计

基础形式：

[
Fitness=f(x)
]

其中：

* (f(x)) 为 surrogate model 输出

---

# E.3 参数约束

必须加入：

| 参数  | 范围   |
| --- | ---- |
| 翼角  | 物理限制 |
| 速度  | 合理区间 |
| DRS | 离散状态 |

防止：

* 无物理意义解
* OOD 搜索

---

# E.4 多场景优化

必须定义多个 scenario：

例如：

| 场景     | 目标   |
| ------ | ---- |
| 高速赛道   | 低阻力  |
| 高下压力赛道 | 高稳定性 |
| 雨战     | 鲁棒稳定 |

要求：

输出不同场景最优参数组合。

---

# 模块 F：改进 PSO（创新模块）

---

# F.1 自适应惯性权重

实现：

[
w_t=w_{max}-(w_{max}-w_{min})\cdot \frac{t}{T}
]

目标：

平衡：

* 全局探索
* 局部开发

---

# F.2 风险敏感 PSO（推荐核心创新）

核心思想：

代理模型存在预测不确定性。

PSO 不应只最大化预测值。

建议适应度：

[
Fitness=\mu(x)-\lambda \sigma(x)
]

其中：

* (\mu(x))：预测均值
* (\sigma(x))：预测不确定性

目标：

避免：

* surrogate hallucination
* 高风险虚假最优解

---

# F.3 数据密度惩罚（可选）

思想：

远离训练分布时降低 fitness。

例如：

[
Fitness=f(x)-\alpha D(x)
]

其中：

* (D(x)) 为分布距离

---

# 模块 G：实验设计

---

# G.1 Baseline 对比实验

比较：

| 方法                 |
| ------------------ |
| Standard PSO       |
| Adaptive PSO       |
| Risk-sensitive PSO |

---

# G.2 多次独立实验

每组：

* 至少运行 20 次

记录：

* 收敛速度
* 最优值
* 方差

---

# G.3 消融实验（Ablation Study）

研究：

* 去掉风险项会怎样
* 去掉约束会怎样
* 去掉自适应权重会怎样

---

# G.4 鲁棒性实验

加入：

* 噪声扰动
* 初始化变化
* 不同随机种子

分析：

算法稳定性。

---

# 模块 H：结果可视化

---

必须生成：

| 图表                    | 内容    |
| --------------------- | ----- |
| Loss Curve            | 训练过程  |
| PSO Convergence Curve | 收敛过程  |
| SHAP Summary Plot     | 特征重要性 |
| Residual Plot         | 残差    |
| Fitness Landscape     | 搜索空间  |
| Scenario Comparison   | 多场景结果 |

---

# 6. 推荐代码结构

```text
project/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── optimization/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
│
├── experiments/
├── outputs/
├── reports/
├── figures/
└── README.md
```

---

# 7. 推荐技术栈

---

## 数据处理

* pandas
* numpy

---

## 可视化

* matplotlib
* seaborn
* plotly

---

## 机器学习

* scikit-learn
* xgboost
* pytorch（可选）

---

## PSO

可选：

* pyswarms
* 自定义实现

推荐：

优先自定义实现。

---

# 8. 最终交付成果

---

# 8.1 模型成果

包括：

* 最终 surrogate model
* 改进 PSO
* 多场景优化系统

---

# 8.2 实验成果

包括：

* 对比实验
* 消融实验
* 鲁棒性实验

---

# 8.3 可视化成果

包括：

* 图表
* 收敛曲线
* SHAP 分析
* Scenario 对比图

---

# 8.4 文档成果

包括：

* README
* 实验报告
* 方法说明
* 结果分析

---

# 9. 项目关键创新点

---

## 创新点 1

基于 surrogate model 的赛车气动快速预测。

---

## 创新点 2

PSO 与代理模型融合。

---

## 创新点 3

多场景条件优化。

---

## 创新点 4（核心）

风险敏感 PSO：

[
Fitness=\mu(x)-\lambda \sigma(x)
]

提升：

* 鲁棒性
* 搜索可信度
* 泛化能力

---

# 10. 风险与应对策略

---

## 风险 1：BP 效果不佳

解决：

* 使用 XGBoost
* 使用 Random Forest
* 使用集成模型

---

## 风险 2：PSO 搜索到 OOD 区域

解决：

* 参数边界约束
* density penalty
* 风险敏感 fitness

---

## 风险 3：PSO 收敛不稳定

解决：

* 自适应惯性权重
* 多次独立实验
* 动态学习因子

---

# 11. 最终研究目标

最终希望实现：

* 高精度 surrogate model
* 高效率参数搜索
* 多工况智能优化
* 具备可信度约束的优化系统

形成：

“实时预测 — 智能寻优 — 多场景决策”

完整技术框架。
