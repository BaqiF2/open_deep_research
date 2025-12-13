# 中文版本： 作为对deep research的学习


# 🔬 Open Deep Research（开放深度研究）

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

深度研究已成为最受欢迎的智能体应用之一。这是一个简单、可配置的、完全开源的深度研究智能体，支持多种模型提供商、搜索工具和MCP服务器。其性能与许多流行的深度研究智能体相当（[详见深度研究排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)）。

<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

### 🔥 最新更新

**2025年8月14日**：查看我们的免费课程[这里](https://academy.langchain.com/courses/deep-research-with-langgraph)（以及课程代码库[这里](https://github.com/langchain-ai/deep_research_from_scratch)）关于构建开放深度研究。

**2025年8月7日**：添加了GPT-5并更新了深度研究基准评估，包含GPT-5结果。

**2025年8月2日**：在[深度研究基准排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)上获得第6名，总体得分为0.4344。

**2025年7月30日**：在我们的[博客文章](https://rlancemartin.github.io/2025/07/30/bitter_lesson/)中阅读关于我们从原始实现到当前版本的演进。

**2025年7月16日**：在我们的[博客](https://blog.langchain.com/open-deep-research/)中阅读更多内容，并观看我们的[视频](https://www.youtube.com/watch?v=agGiWUpxkhg)获取快速概述。

### 🚀 快速开始

1. 克隆代码库并激活虚拟环境：
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # 在Windows上：.venv\Scripts\activate
```

2. 安装依赖：
```bash
uv sync
# 或
uv pip install -r pyproject.toml
```

3. 设置您的`.env`文件以自定义环境变量（用于模型选择、搜索工具和其他配置设置）：
```bash
cp .env.example .env
```

4. 通过本地LangGraph服务器启动智能体：

```bash
# 安装依赖并启动LangGraph服务器
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

这将在您的浏览器中打开LangGraph Studio UI。

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API文档: http://127.0.0.1:2024/docs
```

在`messages`输入字段中提出问题并点击`提交`。在"管理助手"选项卡中选择不同的配置。

### ⚙️ 配置

#### LLM :brain:

Open Deep Research通过[init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)支持广泛的LLM提供商。它为几个不同的任务使用LLM。有关更多详细信息，请参阅[configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py)文件中的以下模型字段。这可以通过LangGraph Studio UI访问。

- **总结**（默认：`openai:gpt-4.1-mini`）：总结搜索API结果
- **研究**（默认：`openai:gpt-4.1`）：驱动搜索智能体
- **压缩**（默认：`openai:gpt-4.1`）：压缩研究结果
- **最终报告模型**（默认：`openai:gpt-4.1`）：编写最终报告

> 注意：所选模型需要支持[结构化输出](https://python.langchain.com/docs/integrations/chat/)和[工具调用](https://python.langchain.com/docs/how_to/tool_calling/)。

> 注意：对于OpenRouter：遵循[此指南](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408)，对于通过Ollama的本地模型，请参阅[设置说明](https://github.com/langchain-ai/open_deep_research/issues/65#issuecomment-2743586318)。

#### 搜索API :mag:

Open Deep Research支持广泛的搜索工具。默认情况下，它使用[Tavily](https://www.tavily.com/)搜索API。具有完整的MCP兼容性，并为Anthropic和OpenAI提供原生Web搜索。有关更多详细信息，请参阅[configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py)文件中的`search_api`和`mcp_config`字段。这可以通过LangGraph Studio UI访问。

#### 其他

请参阅[configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py)中的各种其他设置字段，以自定义Open Deep Research的行为。

### 📊 评估

Open Deep Research配置为使用[深度研究基准](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)进行评估。该基准包含100个博士级研究任务（50个英文，50个中文），由22个领域（例如科学与技术、商业与金融）的领域专家精心制作，以反映真实世界的深度研究需求。它有2个评估指标，但排行榜基于RACE分数。这使用LLM-as-a-judge（Gemini）根据专家编译的黄金标准报告集评估研究报告。

#### 使用方法

> 警告：在100个示例上运行可能会花费约20-100美元，取决于模型选择。

数据集可通过[此链接](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d)在LangSmith上获取。要开始评估，运行以下命令：

```bash
# 在LangSmith数据集上运行综合评估
python tests/run_evaluate.py
```

这将提供一个指向LangSmith实验的链接，该链接将命名为`YOUR_EXPERIMENT_NAME`。完成后，将结果提取为可提交到深度研究基准的JSONL文件。

```bash
python tests/extract_langsmith_data.py --project-name "YOUR_EXPERIMENT_NAME" --model-name "you-model-name" --dataset-name "deep_research_bench"
```

这将创建`tests/expt_results/deep_research_bench_model-name.jsonl`，包含所需格式。将生成的JSONL文件移动到深度研究基准代码库的本地克隆，并遵循他们的[快速开始指南](https://github.com/Ayanami0730/deep_research_bench?tab=readme-ov-file#quick-start)进行评估提交。

#### 结果

| 名称 | 提交 | 总结 | 研究 | 压缩 | 总成本 | 总令牌 | RACE分数 | 实验 |
|------|------|------|------|------|--------|--------|----------|------|
| GPT-5 | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits) | openai:gpt-4.1-mini | openai:gpt-5 | openai:gpt-4.1 |  | 204,640,896 | 0.4943 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| 默认值 | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98 | 58,015,332 | 0.4309 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined) |
| Claude Sonnet 4 | [f877ea9](https://github.com/langchain-ai/open_deep_research/pull/163/commits/f877ea93641680879c420ea991e998b47aab9bcc) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09 | 138,917,050 | 0.4401 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=04f6002d-6080-4759-bcf5-9a52e57449ea&baseline=undefined) |
| 深度研究基准提交 | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83 | 207,005,549 | 0.4344 | [链接](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined) |

### 🚀 部署和使用

#### LangGraph Studio

按照[快速开始](#-快速开始)在本地启动LangGraph服务器并在LangGraph Studio上测试智能体。

#### 托管部署

您可以轻松部署到[LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。

#### Open Agent Platform

Open Agent Platform（OAP）是一个UI，非技术用户可以从中构建和配置自己的智能体。OAP非常适合让用户配置最适合其需求和要解决问题的深度研究器。

我们已经将Open Deep Research部署到我们OAP的公共演示实例。您只需添加API密钥，就可以自己测试深度研究器！在[这里](https://oap.langchain.com)试用。

您也可以部署自己的OAP实例，并在其中让用户使用您自己的自定义智能体（如深度研究器）。
1. [部署Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [将深度研究器添加到OAP](https://docs.oap.langchain.com/setup/agents)

### 旧版实现 🏛️

`src/legacy/`文件夹包含两个早期的实现，它们提供了自动化研究的替代方法。它们比当前实现性能较低，但为理解深度研究的不同方法提供了替代思路。

#### 1. 工作流实现（`legacy/graph.py`）
- **计划和执行**：具有人机交互规划的结构化工作流
- **顺序处理**：逐个创建部分并进行反思
- **交互式控制**：允许反馈和报告计划批准
- **质量导向**：通过迭代优化强调准确性

#### 2. 多智能体实现（`legacy/multi_agent.py`）
- **监督者-研究者架构**：协调的多智能体系统
- **并行处理**：多个研究者同时工作
- **速度优化**：通过并发实现更快的报告生成
- **MCP支持**：广泛的模型上下文协议集成