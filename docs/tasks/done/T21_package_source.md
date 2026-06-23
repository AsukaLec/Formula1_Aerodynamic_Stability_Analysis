# T21 — 打包源代码到 source_code/

## 描述
将"真正需要的源代码"打包到根目录新文件夹 `source_code/`，供提交。最小源代码方案 + 结果图片，不含数据/模型/报告 tex。

## 用户决策
- 范围：仅源代码（最小）= src + experiments + requirements + README。
- 图片：只放图片（report_v2/figures 的 32 张最终 PNG）→ source_code/figures/。
- 文件夹名：source_code/。

## 交付结构
- src/（去 __pycache__）
- experiments/（去 __pycache__）
- figures/（32 张最终结果图）
- requirements.txt（+pytorch-tabnet）
- README.md（新写）

## 关键处理
- 3 个新脚本（ood/weight/desaturation）包内副本：输出路径由 ../report_v2/figures 改为 ../figures（自洽）。
- 不动原仓库；仅在 source_code/ 内操作。

## 不收
data/、outputs/、report_v2 tex、latex_report/、reports/、docs/、ppt_materials/、others/、v1_related/、.git/、根目录草稿 md。

## 状态
- [x] 复制 src/experiments（去 __pycache__）+ figures（32 张）
- [x] requirements.txt（+pytorch-tabnet）
- [x] 修正 3 脚本输出路径（../report_v2/figures → ../figures，含 docstring）
- [x] README.md（中文：简介/结构/环境/数据获取/运行顺序/脚本产物表）
- [x] 校验：无 report_v2 残留引用、py_compile 通过、清空 __pycache__

## 结果
`source_code/`：README.md + requirements.txt + src/ + experiments/(16) + figures/(32 PNG)，
共 50 个 .py、总计 5.5M。仅源代码 + 结果图，不含数据/模型/报告 tex。原仓库未改动。
