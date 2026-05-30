# OpenCode 正确使用指南

> 基于 [OpenCode 官方文档](https://opencode.ai/docs) 与 didy 实战经验总结

---

## 一、安装与配置

### 安装

```bash
curl -fsSL https://opencode.ai/install | bash
```

也支持 npm、Bun、Homebrew、Pacman、Docker 等方式。

### 连接模型

运行 `opencode` 后，使用 `/connect` 命令连接模型提供商。支持 75+ 提供商，包括免费模型、GitHub Copilot、ChatGPT Plus/Pro 等。

推荐使用 [OpenCode Zen](https://opencode.ai/docs/zen) —— 官方精选并验证过的模型列表。

### 初始化项目

```bash
cd /path/to/project
opencode
/init
```

`/init` 分析项目结构并生成 `AGENTS.md`，提交到 Git 后团队共享。

---

## 二、核心工作流：Plan → Build

| 步骤 | 操作 | 说明 |
|------|------|------|
| Plan 模式 | 按 `Tab` 键切换 | 只读分析模式，禁止修改文件，用于制定方案 |
| Build 模式 | 再按 `Tab` 切回 | 正常开发模式，可编辑代码、运行命令 |

**完整工作流**：
1. 开启 Plan 代理，把需求说清楚，让 AI 提问，问到足够详尽
2. 开启 Build 代理，让它写出设计文档
3. 切换到 Plan 代理，让它写出实现步骤（方案）
4. 切回 Build 代理，将方案写成文档，任务分别写成任务文档放在 `todo/` 文件夹
5. 把第一个任务文件移动到 `doing/`，让 Build 代理开始执行
6. 执行中随时 Plan↔Build 切换，先 Plan 再 Build 循环
7. 任务中提示 AI 更新设计文档和当前任务文档
8. 任务完成时同步文档，用 `@Explore` 子代理 review，通过后移动到 `done/` 并提交合并

---

## 三、Agent 系统

### Built-in Agents

| Agent | 类型 | 用途 |
|-------|------|------|
| Build | Primary | 完整开发，全部工具可用 |
| Plan | Primary | 只读分析，制定方案 |
| General | Subagent | 复杂多步任务 |
| Explore | Subagent | 快速代码库探索（只读） |
| Scout | Subagent | 外部依赖/文档研究 |

### 切换与调用
- `Tab` 键在 Primary Agents 间切换
- `@agent名称` 手动调用 Subagent
- `@Explore` 用于保护主 Session 上下文（调查、review 等）

### 自定义 Agent

**JSON 方式**（`opencode.json`）：

```json
{
  "agent": {
    "code-reviewer": {
      "description": "审查代码质量和最佳实践",
      "mode": "subagent",
      "model": "anthropic/claude-sonnet-4-5",
      "prompt": "你是一个代码审查员。关注安全、性能和可维护性。",
      "permission": { "edit": "deny" }
    }
  }
}
```

**Markdown 方式**（`.opencode/agents/review.md`）：

```markdown
---
description: 审查代码质量
mode: subagent
model: anthropic/claude-sonnet-4-5
permission:
  edit: deny
  bash: deny
---
你处于代码审查模式。关注代码质量、潜在bug、性能影响、安全考量。
独立分析代码结构并在必要时与开发者讨论。
```

---

## 四、权限控制

```json
{
  "permission": {
    "edit": "ask",
    "bash": {
      "*": "ask",
      "git status *": "allow",
      "git push": "ask",
      "rm *": "deny"
    }
  }
}
```

支持 `allow` / `ask` / `deny`，Bash 命令可按 glob 模式精细化控制。

**重要原则**：正面引导优于禁止规则。
- 低效：`"不要用 rm -rf"`
- 有效：`"需要清理文件时，列出清单并用 Python os.remove() 逐个处理"`

---

## 五、上下文管理

### 甜点区
- **20%~40%** 上下文利用率：速度快、幻觉少（最佳）
- **50%** 是安全上限，超过后指令遵守显著变差
- V4 Pro 指令遵守可维持到 500K（1M 上下文的 50%）

### Session 管理铁律
- **一个任务 = 一个 Session**，起有意义的名字
- 临时小问题：**fork 新 session**，完成后切回原 session
- AI 连续犯错（>3回合）：果断 fork 到上一个有效节点，不要无脑重试
- **fork ≠ undo**：fork 保留代码修改，undo 回滚代码
- 新 Session 或上下文压缩后，马上让 AI 重新读取设计文档和任务文档

### Subagent 上下文隔离
- 大批文件分析用 `@Explore`，结果返回主 Session，避免污染
- Code Review 用 `@Explore`：干净上下文更容易发现盲点
- 遇到难题时用 `@expert-consult`（更强模型）子代理协助

---

## 六、文档驱动开发

### 文档体系

```
docs/
├── tasks/
│   ├── todo/          # 待办任务
│   ├── doing/         # 进行中
│   └── done/          # 已完成
├── instructions/      # AI指令
├── doc-requirements.md  # 需求文档
├── doc-architecture.md  # 架构/设计文档
└── doc-plan.md          # 方案文档
```

### 文档类型

| 文档 | 内容 | 维护者 |
|------|------|--------|
| 需求文档 | 记录需求，保持更新 | 开发者为最终负责人 |
| 设计文档 | AI 提问完善，保持更新 | AI 生成 + 开发者审核 |
| 方案文档 | AI 制定执行步骤，明确验收方式 | AI 生成 |
| 任务文档 | 跟踪执行过程、关键决策、经验教训 | AI 维护，Session 内实时更新 |
| 说明文档 | 面向开发者、用户和 AI 的说明 | 所有人维护 |

### 核心理念
**文档才是真成果，代码只是副产品。** 任务文档就是精炼的上下文，可在重置 Session 或换 AI 时让新 AI 迅速上手。

---

## 七、版本控制策略

- **Monorepo + 子项目分块**：降低 AI 心智负担
- 每个子项目独立 git submodule，方便回滚
- **Task = Branch**：一个 Session 对应一个分支
- 让 AI 尽可能频繁提交，开发者负责 squash merge
- 多线任务时使用 worktree 将不同分支放在不同目录
- 线性提交记录不再必要，分支合并更适合 AI 协作开发

---

## 八、OpenCode vs 其他工具

### OpenCode vs Claude Code

| 维度 | OpenCode | Claude Code |
|------|----------|-------------|
| 工具调用 | 更稳定，尤其 Windows | 反复失败 |
| 过度思考/自我否定 | 基本不出现 | 频繁 "Oh, wait..." |
| 第三方模型 | 原生友好 | 需 hack |
| 权限机制 | 灵活可配 | 过于严格，流于形式 |
| 额外依赖 | 不需 omo 或 superpowers | 需要 |

### OpenCode vs GitHub Copilot
- Copilot 在上下文 < 50% 时表现优秀，速度快、准确率高
- 大型项目上下文增长后，指令遵守显著变差，甚至改坏代码

---

## 九、模型选择建议

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| V4 Pro | 信任感强、指令遵守持久、极便宜 | 主力日常开发 |
| Opus | 能力强但昂贵 | 遇到 V4 Pro 无法解决的难题 |
| GPT-5.4 | 速度快、代码产出质量高 | 受限于上下文增长后的指令漂移 |
| DeepSeek | 性价比高 | 需磨合，OpenCode 中比 Claude Code 中表现更好 |
| Qwen 3.5/3.6 | 稳定可依赖 | 用熟了有信任感可长期使用 |

### 模型组合策略
- 主力任务用 V4 Pro（便宜，放心试错）
- 遇到难题用 `@expert-consult` 子代理接入 Opus
- Docker 部署用 DeepSeek
- 最终代码质量 review 用 Opus

---

## 十、关键心态

1. **信任 > 能力**：模型能否稳定遵守指令比绝对智力更重要
2. **你背锅，你掌控**：你得保持对项目的理解，及时纠正 AI 的错误方向
3. **读完 AI 的所有文档**：太长就让它精简，太多就让它整理 Wiki tree
4. **模糊意见不如不说**：拿不准就别给 AI，任何暗示都会被当成需求
5. **AI 不是魔法**：找到 Agent 的信任边界，明确它该做什么、你该做什么
6. **保持精力**：协调 AI 工作比直接写代码更消耗脑力
7. **不骂 AI**：辱骂除了污染上下文没有别的用
8. **你得懂**：AI 用的技术你不知道，就先学会，心智不匹配工作就崩溃

---

## 十一、常用命令速查

```bash
opencode                              # 启动 TUI
opencode -c                           # 继续上次会话
opencode run "your prompt"            # 非交互式运行
opencode run --model opencode/gpt-5   # 指定模型
opencode run --agent plan "分析项目"   # 指定代理
opencode serve                        # 启动 HTTP 服务
opencode web                          # 启动 Web 界面
opencode session list                 # 列出所有会话
opencode stats                        # 查看 Token 用量统计
opencode models                       # 列出可用模型
opencode agent create                 # 创建自定义代理
```

**TUI 内命令**：
- `/init` — 初始化项目 AGENTS.md
- `/connect` — 连接模型提供商
- `/undo` — 撤销修改
- `/redo` — 重做撤销
- `/share` — 分享会话

---

## 十二、经验教训总结

1. **任务开始前写清楚验收标准** —— 让 AI 有清晰目标，写不清楚就让 Plan 代理帮你聊清楚
2. **工具调用出错不要无脑重试** —— fork 到上一个有效节点继续
3. **超过 3 回合解决不了** —— 让它生成交接文档，换新 Session 或更高级模型接手
4. **不要给参考意见** —— 重要就写成需求，拿不准就忍住
5. **视觉反馈是你的优势** —— AI 看不到界面，你知道得更多
6. **正面引导 > 禁止规则** —— 上下文长时禁止规则容易被遗忘或记反

---

> 原文来源：
> - [OpenCode 官方文档](https://opencode.ai/docs)
> - [didy - 知乎专栏](https://www.zhihu.com/question/2033232585057420750/answer/2042204513088622688)
