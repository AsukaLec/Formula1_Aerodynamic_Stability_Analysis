我不能保证自动读取——每个新 Session 我都是全新实例，没有之前对话的记忆。虽然 AGENTS.md 在仓库里，但我不一定会主动打开它。
根据 OpenCode 使用指南的建议：**"新 Session 或上下文压缩后，马上让 AI 重新读取设计文档和任务文档"**。

每次新 Session 开始时，你只需说一句：

**请先阅读 AGENTS.md 和 docs/ 下的核心文档，然后继续。**


这一句话会触发我依次读取 AGENTS.md → doc-plan.md → doc-requirements.md → doc-architecture.md，以及 docs/tasks/doing/ 中当前正在进行的任务，完整恢复上下文。