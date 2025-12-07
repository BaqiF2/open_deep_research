# Open Deep Research 仓库概览

## 项目描述
Open Deep Research 是一个可配置的、完全开源的深度研究代理，支持多个模型提供商、搜索工具和 MCP（模型上下文协议）服务器。它能够实现自动化研究，支持并行处理和全面的报告生成。

## 仓库结构

### 根目录
- `README.md` - 完整的项目文档，包含快速入门指南
- `pyproject.toml` - Python 项目配置和依赖项
- `langgraph.json` - LangGraph 配置，定义主图入口点
- `uv.lock` - UV 包管理器锁文件
- `LICENSE` - MIT 许可证
- `.env.example` - 环境变量模板（不跟踪）

### 核心实现 (`src/open_deep_research/`)
- `deep_researcher.py` - 主要的 LangGraph 实现（入口点：`deep_researcher`）
- `configuration.py` - 配置管理和设置
- `state.py` - 图状态定义和数据结构
- `prompts.py` - 系统提示词和提示模板
- `utils.py` - 工具函数和辅助函数
- `files/` - 研究输出和示例文件

### 遗留实现 (`src/legacy/`)
包含两个早期的研究实现：
- `graph.py` - 计划-执行工作流，支持人工干预
- `multi_agent.py` - 监督者-研究者多智能体架构
- `legacy.md` - 遗留实现的文档
- `CLAUDE.md` - 遗留特定的 Claude 指令
- `tests/` - 遗留特定的测试

### 安全 (`src/security/`)
- `auth.py` - LangGraph 部署的身份验证处理器

### 测试 (`tests/`)
- `run_evaluate.py` - 主评估脚本，配置为在深度研究基准上运行
- `evaluators.py` - 专门的评估函数
- `prompts.py` - 评估提示词和标准
- `pairwise_evaluation.py` - 比较评估工具
- `supervisor_parallel_evaluation.py` - 多线程评估

### 示例 (`examples/`)
- `arxiv.md` - ArXiv 研究示例
- `pubmed.md` - PubMed 研究示例
- `inference-market.md` - 推理市场分析示例

## 关键技术
- **LangGraph** - 工作流编排和图执行
- **LangChain** - LLM 集成和工具调用
- **多个 LLM 提供商** - 支持 OpenAI、Anthropic、Google、Groq、DeepSeek
- **搜索 API** - Tavily、OpenAI/Anthropic 原生搜索、DuckDuckGo、Exa
- **MCP 服务器** - 模型上下文协议，用于扩展能力

## 开发命令
- `uvx langgraph dev` - 启动开发服务器（包含 LangGraph Studio）
- `python tests/run_evaluate.py` - 运行全面评估
- `ruff check` - 代码检查
- `mypy` - 类型检查

## 配置
所有设置可通过以下方式配置：
- 环境变量（`.env` 文件）
- LangGraph Studio 中的 Web UI
- 直接修改配置

关键设置包括模型选择、搜索 API 选择、并发限制和 MCP 服务器配置。