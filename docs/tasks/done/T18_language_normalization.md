# T18 — report_v2 学术语言规范化

## 描述

按 `redo_target2.md` 审查意见，对 `report_v2/` 做学术写作规范化修订（不增实验、不改数据/图/章节结构与结论），提升正式性、克制性与客观性。

## 用户决策

- 章节问题框：**中性化保留**（"本章回答：…？" → 陈述式"本章研究的问题：…"，去反问尾注）。
- P2（课程契合段 + 导览去剧透）：**本轮一并做**。
- 章号交叉引用错位 2 的修正：**改用 `\label/\ref` 自动编号**。

## 执行清单

1. 章号引用：02–11 每个 `\section` 加 `\label{sec:...}`；正文 `第N章`→`第\ref{sec:...}章`；被引子节加 `\label{subsec:...}`，`§N.x`→`\ref{subsec:...}`。
2. 问题框中性化保留。
3. P0：删叙事化/口语化词（故事/第一幕/转折/高潮/伏笔/胡说/心虚/炫耀/纹丝不动/殊途同归/侧证/无济于事…）；Core Claim 仅保留摘要/引言/结论。
4. P1：摘要 −30%；弱化 Landscape 唯一解释色彩（→"当前最有说服力的解释/一致性解释框架"）；`证明/证伪/必然`→`支持/表明/指向/与…一致`。
5. P2：引言加《本研究与计算智能课程主题的关系》短段；§2.4 导览去剧透。
6. 静态校验；保持 Overleaf（XeLaTeX）可编译。

## \label 命名

sec:intro / dataset / surrogate / anomaly / uncertainty / risk / refine / converge / landscape / conclusion；
subsec:desat / counterfactual / wing-sens / data-struct / shap。

## 状态

- [x] 章号 label/ref（10 个 sec + 5 个 subsec，全部 \ref 解析；正文 `第N章/§N.x` 清零）
- [x] 问题框中性化（"本章回答…？"→"本章研究的问题：…"）
- [x] P0 去叙事/口语 + Core Claim 频率（仅留摘要/引言/结论）
- [x] P1 摘要压缩（约 −35%）+ Landscape 弱化（→"一致性解释/当前最有说服力的解释"）+ 措辞强度（证明/证伪/必然→表明/削弱/支持）
- [x] P2 课程契合段（引言新增小节）+ 导览去剧透（合并式结构表）
- [x] 静态校验：环境/括号平衡、\ref↔\label 全解析、15 张图均就位

## 编译说明

本机无 TeX 工具链，未做本地编译。仅改措辞与交叉引用，preamble 不变，
`report_v2/` 仍可原封不动上传 Overleaf（XeLaTeX）。
