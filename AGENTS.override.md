# 本仓中央文档入口

开始前必须完整读取 [本仓原有规则](AGENTS.md)，继续遵守其中的命令、约束及更细目录规则。本文件只增加工作区导航，不替代或放宽原有规则。

先读以下中央文档，再修改代码：

- [统一入口](../../../docs/README.md)、[AI 工作流](../../../docs/engineering/ai-workflow.md)、[文档归属 ADR](../../../docs/architecture/decisions/ADR-0002-central-authored-documentation.md)。
- [仓库地图](../../../docs/architecture/repository-map.md)、[工具边界](../../../docs/architecture/tool-boundaries.md)。
- 本仓模块：[research-orchestration](../../../docs/modules/research-orchestration/README.md)、[models](../../../docs/modules/models/README.md)；修改前读对应不变量。
- [工单指引](../../../docs/tracking/README.md)、[统一看板](../../../docs/tracking/board.md)、[ticket 模板](../../../docs/templates/ticket.md)。从已有 ticket 开始，交付时更新同一工单及验证证据。

自编架构、实验、优化、排障和运维记录在上述中央 docs 中维护；本仓保留源码、可执行 schema、测试及上游原有文档。实际执行的验证与建议执行的验证必须分开记录。

链接以本文件所在仓库根为基准；规范布局下工作区根为 ../../..。若单独克隆到其他位置，先定位工作区管理仓库，或读取操作者已设置的 WORKSPACE_DOCS_ROOT；中央文件不可读时明确报告缺口，不新造另一套规范、不声称已读。

入口维护与核验见 [接入记录](../../../docs/governance/repository-ai-entrypoints.md)。只执行当前改动所需的仓库检查；发布另按既有授权流程。
