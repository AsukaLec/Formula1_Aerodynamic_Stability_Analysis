# Blueprint Final Review Feedback

## 总体评价

当前 Blueprint 已经达到可执行水平。

可以认为已经通过：

* 研究问题重构
* Story Arc 重构
* 实验重新定位
* 章节因果关系重构

四个核心检查项。

因此：

> 允许进入正式重构阶段。

但是，在开始修改正文之前，建议先完成以下最后一轮 Blueprint 修订。

这些修改成本很低，但会显著提升最终报告质量。

---

# 1. 增加 Core Claim（全文唯一主张）

## 当前问题

Blueprint 中已经明确了：

* Research Question
* Story Arc
* Contributions

但缺少：

> 全文唯一核心主张（Core Claim）

这会导致后续写作过程中再次出现内容发散。

---

## 建议补充

新增一个独立部分：

### Core Claim

本文最核心的主张是：

> 在代理模型驱动优化中，
>
> 数据覆盖度（Data Coverage）、
> 不确定性管理（Uncertainty Management）、
> 优化景观结构（Optimization Landscape）
>
> 比优化器选择本身更决定最终优化质量。

---

要求：

全文所有实验必须服务于这个主张。

如果某个实验无法支撑该主张：

* 降级
* 压缩
* 删除

---

# 2. 严格区分 Contribution 与 Finding

## 当前问题

Blueprint 中存在轻微混淆：

Landscape Diagnosis 被写入 Contribution。

这是不严谨的。

---

## 原因

Contribution：

表示研究团队做了什么。

Finding：

表示研究团队发现了什么。

二者不能混用。

---

## 当前写法（不推荐）

Contribution 4：

Landscape Diagnosis 揭示：

数据结构与景观退化而非优化器主导结果。

---

## 建议修改

### Contribution 4

构建并验证了一套：

异常发现

↓

不确定性分析

↓

数据精炼

↓

景观诊断

↓

根因定位

的可信代理优化诊断流程。

---

然后单独增加：

### Key Finding

景观退化与数据结构限制是导致优化结果趋同的重要原因。

---

Landscape Diagnosis 应该支撑 Finding。

而不是直接成为 Contribution。

---

# 3. Landscape Diagnosis 从“贡献”降级为“发现”

## 当前问题

Blueprint 中仍有轻微倾向：

将 Landscape Diagnosis 视为最终答案。

---

## 风险

Landscape 是一种解释机制。

未来可能被：

* 数据覆盖不足
* OOD效应
* 数据流形约束
* 共线性结构

等其它机制部分替代。

---

因此：

### Trustworthy Optimization

应该是主线。

### Landscape Diagnosis

应该是当前版本最有说服力的解释。

---

建议：

保持 Landscape 的高潮地位。

但不要让其成为全文唯一支柱。

---

# 4. 增加“章节问题表”

## 当前问题

Blueprint 已有章节结构。

但没有明确说明：

> 每个章节究竟在回答什么问题？

---

## 建议新增

### Chapter 2

回答：

为什么我们开始怀疑代理模型？

---

### Chapter 3

回答：

代理模型是否可信？

---

### Chapter 4

回答：

如何避免优化器利用代理盲区？

---

### Chapter 5

回答：

这些盲区是否真实存在？

---

### Chapter 6

回答：

为什么不同优化器和不同场景得到相似解？

---

### Chapter 7

回答：

这些异常现象是否存在统一根因？

---

### Chapter 8

回答：

可信代理优化框架能够带来什么工程启示？

---

这样能够确保：

每章都有明确任务。

不会退化成实验堆砌。

---

# 5. 明确 Porpoising 章节定位

## 当前问题

Blueprint 中：

Porpoising 部分仍处于待定状态。

---

## 建议

保留。

但降级为：

### Case Study

而非主线章节。

---

## 原因

Porpoising 主要证明：

框架具有应用价值。

它回答的是：

> Framework Application

而不是：

> Trustworthiness

---

因此：

可以保留。

但不要承担主线推进任务。

---

建议放置于：

最终讨论章节。

或者案例研究章节。

---

# 6. 重新调整新增实验优先级

## 当前问题

Blueprint 中：

去饱和实验（Desaturation Experiment）

被列为最高优先级。

---

## 风险

这是一个高风险实验。

因为存在以下可能：

### 情况A

解发生明显变化

支持景观退化假设。

很好。

---

### 情况B

解没有明显变化

则会削弱当前叙事。

---

因此：

不应作为第一优先级。

---

## 推荐优先级

### Priority A

#### OOD Pressure Test

目的：

验证风险敏感机制是否能够识别代理模型幻觉。

---

#### Weight Sensitivity Analysis

目的：

验证目标权重是否真正影响解。

---

这两个实验：

无论结果如何都具有解释价值。

---

### Priority B

#### Desaturation Experiment

作为：

景观退化机制的强化验证。

---

而非主线成立的必要条件。

---

# 最终结论

当前 Blueprint 已通过审核。

允许进入正式重构阶段。

但建议在开始修改正文之前：

完成以下六项修订：

1. 增加 Core Claim；
2. 区分 Contribution 与 Finding；
3. Landscape 从 Contribution 调整为 Finding；
4. 增加章节问题表；
5. 明确 Porpoising 为 Case Study；
6. 调整新增实验优先级。

完成后即可开始正式重构报告。
