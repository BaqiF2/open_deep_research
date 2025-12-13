"""深度研究代理的主要LangGraph实现。

这是一个完整的深度研究工作流程，使用LangGraph构建多步骤研究系统。
主要功能包括：用户澄清、研究计划、研究执行、报告生成等阶段。
"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_base_url_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[
    Literal["write_research_brief", "__end__"]]:
    """分析用户消息，如果研究范围不清晰则询问澄清问题。

    这是研究流程的第一道关卡。LangGraph会将用户输入的原始消息传递给此函数，
    该函数决定是否需要进行澄清，或者可以直接进入研究阶段。

    核心逻辑：
    1. 检查配置中是否启用了澄清功能（allow_clarification）
    2. 如果启用，使用AI分析用户消息是否足够清晰
    3. 如果不清晰，返回澄清问题给用户
    4. 如果清晰，直接进入研究简报阶段

    Args:
        state: 包含用户消息的当前代理状态。在LangGraph中，state是跨节点共享的数据容器
        config: 运行时配置，包含模型设置和用户偏好。LangGraph通过config传递外部参数

    Returns:
        Command: LangGraph中的命令对象，控制流程走向：
                - goto: 指定下一个要执行的节点
                - update: 可选，更新状态数据
                如果需要澄清，则结束流程（goto=END）等待用户回复
                如果不需要澄清，则进入write_research_brief阶段
    """
    # 步骤1：检查配置中是否启用了澄清功能
    configurable = Configuration.from_runnable_config(config)
    # 默认的配置是TRUE
    if not configurable.allow_clarification:
        # 跳过澄清步骤，直接进入研究。这是快速通道，适合明确的研究请求
        return Command(goto="write_research_brief")

    # 步骤2：为结构化澄清分析准备模型
    # 这里使用了结构化输出（with_structured_output），确保AI返回的数据符合ClarifyWithUser格式
    messages = state["messages"]

    # 配置模型，包含结构化输出和重试逻辑
    # .with_structured_output()让AI返回JSON格式的结构化数据
    # .with_retry()在API调用失败时自动重试
    clarification_model = ((
                               ChatOpenAI(model=configurable.research_model,
                                          max_tokens=configurable.research_model_max_tokens,
                                          api_key=get_api_key_for_model(configurable.research_model, config),
                                          base_url=get_base_url_for_model(configurable.research_model),
                                          tags=["langsmith:nostream"])
                           ).with_structured_output(ClarifyWithUser)
                           .with_retry(stop_after_attempt=configurable.max_structured_output_retries))

    # 步骤3：分析是否需要澄清
    # 使用预定义的提示模板分析用户消息是否需要澄清
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # 步骤4：根据澄清分析进行路由
    # FIXME 这个阶段主要更新的状态数据是messages，包含用户消息和AI的问答和最终的研究论点
    if response.need_clarification:
        # 以澄清问题结束，返回给用户。在LangGraph中，END表示流程结束
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # 继续研究，附带验证消息。让用户知道AI理解了他的需求
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """将用户消息转换为结构化研究简报并初始化监督者。

    这是研究流程的第二阶段，负责将用户模糊的需求转化为清晰、可执行的研究计划。

    工作流程：
    1. 接收用户原始消息
    2. 使用AI分析并生成结构化的研究简报（ResearchQuestion对象）
    3. 创建监督者的系统提示，包含研究策略和限制
    4. 将研究简报传递给监督者，开始真正的研究工作

    为什么要这样做？
    - 用户通常只会说"研究AI的发展趋势"，但AI需要知道具体要研究什么方面
    - 研究简报明确了研究范围、深度和重点
    - 监督者会根据这个简报来决定如何分配研究任务

    Args:
        state: 包含用户消息的当前代理状态
        config: 运行时配置，包含模型设置

    Returns:
        Command: 进入研究监督者阶段，附带初始化的上下文
                - research_brief: 结构化的研究计划
                - supervisor_messages: 监督者的初始对话（系统提示+研究简报）
    """
    # 步骤1：为结构化输出设置研究模型
    configurable = Configuration.from_runnable_config(config)
    research_model = ((
                          ChatOpenAI(model=configurable.research_model,
                                     max_tokens=configurable.research_model_max_tokens,
                                     api_key=get_api_key_for_model(configurable.research_model, config),
                                     base_url=get_base_url_for_model(configurable.research_model),
                                     tags=["langsmith:nostream"])
                      ).with_structured_output(ResearchQuestion)
                      .with_retry(stop_after_attempt=configurable.max_structured_output_retries))

    # 步骤2：从用户消息生成结构化研究简报
    # 使用提示模板将用户消息转换为研究简报
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # 步骤3：使用研究简报和指令初始化监督者
    # 监督者需要知道：当前日期、并发研究单元限制、迭代次数限制等
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )

    # 1. 进入监督者阶段时，将研究简报作为初始消息
    # 2. 更新AgentState数据，保存研究简报和监督者消息列表
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            # 使用"override"类型，表示完全替换之前的消息
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """领导研究监督者，规划研究策略并委派给研究人员。

    这是研究流程的核心控制节点。监督者就像项目经理，负责：
    1. 分析研究简报，制定研究策略
    2. 将大任务分解为小任务
    3. 委派给多个研究人员并行执行
    4. 监控研究进度，决定何时结束

    关键概念：
    - think_tool: 监督者可以"思考"，进行战略规划
    - ConductResearch: 将具体研究任务委派给子研究人员
    - ResearchComplete: 宣布研究阶段完成

    LangGraph工作原理：
    - supervisor()生成决策和工具调用
    - supervisor_tools()执行这些工具调用
    - 然后回到supervisor()检查结果并做出下一步决策
    这形成了supervisor -> supervisor_tools -> supervisor的循环

    Args:
        state: 当前监督者状态，包含消息和研究上下文
        config: 运行时配置，包含模型设置

    Returns:
        Command: 继续到supervisor_tools执行工具
                - supervisor_messages: 更新监督者的对话历史
                - research_iterations: 增加迭代计数，防止无限循环
    """
    # 步骤1：使用可用工具配置监督者模型
    configurable = Configuration.from_runnable_config(config)
    # bind_tools将这些工具"绑定"到模型上，AI可以调用它们
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    # 配置模型，包含工具、重试逻辑和模型设置
    research_model = ((
                          ChatOpenAI(model=configurable.research_model,
                                     max_tokens=configurable.research_model_max_tokens,
                                     api_key=get_api_key_for_model(configurable.research_model, config),
                                     base_url=get_base_url_for_model(configurable.research_model),
                                     tags=["langsmith:nostream"])
                      ).bind_tools(lead_researcher_tools)
                      .with_retry(stop_after_attempt=configurable.max_structured_output_retries))

    # 步骤2：基于当前上下文生成监督者响应
    # 监督者会查看之前的对话和系统提示，决定下一步行动 FIXME 上一步由write_research_brief生成的内容
    supervisor_messages = state.get("supervisor_messages", [])
    # 调用大模型得到下一步要执行的内容，继续新主题研究，思考，或者完成
    response = await research_model.ainvoke(supervisor_messages)

    # 步骤3：更新状态并继续执行工具
    # 注意：这里只是更新状态，真正的工具执行在supervisor_tools中
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """执行监督者调用的工具，包括研究委派和战略思考。

    这是工具执行节点，负责处理supervisor()生成的所有工具调用。

    处理三种类型的工具调用：
    1. think_tool - 战略反思，只是记录思考内容，继续对话
    2. ConductResearch - 将研究任务委派给子研究人员（关键功能）
    3. ResearchComplete - 表示研究阶段完成，结束整个研究循环

    并发执行：
    - 多个ConductResearch任务可以并行执行，提高效率
    - 但有最大并发数限制，防止资源耗尽
    - 超过限制的任务会返回错误消息

    退出条件：
    - 超过最大迭代次数
    - 没有工具调用（AI停止决策）
    - 显式调用ResearchComplete

    Args:
        state: 当前监督者状态，包含消息和迭代计数
        config: 运行时配置，包含研究限制和模型设置

    Returns:
        Command: 要么继续监督循环，要么结束研究阶段
    """
    # 步骤1：提取当前状态并检查退出条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 定义研究阶段的退出标准
    # 1. 超过最大迭代次数
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    # 2. 没有工具调用（AI停止决策）
    no_tool_calls = not most_recent_message.tool_calls
    # 3. 显式调用ResearchComplete
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果满足任何终止条件，则退出
    # 进入报告生成阶段，携带所有研究笔记
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                # 获取所有工具内容，加入到笔记列表
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # 步骤2：一起处理所有工具调用（包括think_tool和ConductResearch）
    # 注意：这里不直接执行工具，而是准备工具消息
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # 处理think_tool调用（战略反思）
    # think_tool只是记录AI的思考，不执行实际研究
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    # 如果有反思工具调用，将反思内容加入工具消息中
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"反思已记录: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # 处理ConductResearch调用（研究委派）
    # 这是核心功能：取出所有研究主题
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    # 将研究任务委派给子研究人员
    if conduct_research_calls:
        try:
            # 限制并发研究单元数量，防止资源耗尽
            # 这就像控制同时进行的项目数量，避免团队过载
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            research_tasks = [
                # FIXME 调用研究子图研究，进入到研究子图
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            # 得到研究结果消息，会生成一个ToolMessage
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research",
                                            "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            # 处理溢出的研究调用，返回错误消息
            # 告知用户超过了并发限制，需要减少研究单元数量 FIXME 这个没必要带给模型吧
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            # 聚合所有研究结果的原始笔记
            # 这些笔记将用于最终报告生成
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            # 处理研究执行错误
            # 如果是token限制错误或其他错误，结束研究阶段
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token限制超出或其他错误 - 结束研究阶段
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )

    # 步骤3：返回命令，包含所有工具结果
    # 这些工具消息会被传递给supervisor，继续下一轮决策
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """具体研究人员，对特定主题进行深入研究。

    这是LangGraph中的子图，代表一个具体的研究人员角色。
    监督者可以将不同的研究任务分配给不同的人员（实例）并行执行。

    核心职责：
    1. 接收监督者分配的特定研究主题
    2. 使用可用工具（搜索、思考工具、MCP工具）收集信息
    3. 可以使用think_tool在搜索之间进行战略规划
    4. 不断迭代直到完成研究或达到限制

    工作流程：
    1. 加载配置并验证工具可用性
    2. 配置研究人员模型，绑定可用工具
    3. 生成响应，包含系统上下文
    4. 更新状态并进入工具执行阶段

    工具类型：
    - 搜索工具：获取外部信息
    - think_tool：进行战略思考
    - MCP工具：扩展功能

    Args:
        state: 当前研究人员状态，包含消息和主题上下文
        config: 运行时配置，包含模型设置和工具可用性

    Returns:
        Command: 继续到researcher_tools执行工具
    """
    # 步骤1：加载配置并验证工具可用性
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # 获取所有可用的研究工具（搜索、MCP、think_tool）
    # 如果没有工具，就无法进行研究，这是致命错误
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )

    # 步骤2：配置研究人员模型和工具
    # 准备系统提示，如果可用则包含MCP上下文
    # MCP（Model Context Protocol）允许集成外部工具和服务
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "",
        date=get_today_str()
    )

    # 配置模型，包含工具、重试逻辑和设置
    # bind_tools将所有可用工具绑定到模型上
    research_model = ((
                          ChatOpenAI(model=configurable.research_model,
                                     max_tokens=configurable.research_model_max_tokens,
                                     api_key=get_api_key_for_model(configurable.research_model, config),
                                     base_url=get_base_url_for_model(configurable.research_model),
                                     tags=["langsmith:nostream"])
                      ).bind_tools(tools)
                      .with_retry(stop_after_attempt=configurable.max_structured_output_retries))

    # 步骤3：生成研究人员响应，包含系统上下文
    # 系统消息告诉AI如何行动，人类消息是具体的研究任务
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # 步骤4：更新状态并继续执行工具
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


# 工具执行辅助函数
async def execute_tool_safely(tool, args, config):
    """安全地执行工具，处理错误。

    这是一个包装器，确保工具执行失败不会崩溃整个系统。
    如果工具执行失败，会返回错误消息而不是抛出异常。

    Args:
        tool: 要执行的工具
        args: 工具参数
        config: 运行时配置

    Returns:
        工具执行结果或错误消息字符串
    """
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[
    Literal["researcher", "compress_research"]]:
    """执行研究人员调用的工具，包括搜索工具和战略思考。

    这是研究人员子图中的工具执行节点，类似于supervisor_tools。

    处理多种类型的工具调用：
    1. think_tool - 战略反思，继续研究对话
    2. 搜索工具（tavily_search, web_search） - 信息收集
    3. MCP工具 - 外部工具集成
    4. ResearchComplete - 表示单个研究任务完成

    退出条件：
    - 早期退出：如果没有工具调用（包括原生网络搜索）
    - 晚期退出：达到最大工具调用次数或显式调用ResearchComplete

    为什么需要早期和晚期检查？
    - 早期检查：防止不必要的处理
    - 晚期检查：在处理工具后确认是否继续

    Args:
        state: 当前研究人员状态，包含消息和迭代计数
        config: 运行时配置，包含研究限制和工具设置

    Returns:
        Command: 要么继续研究循环，要么进入压缩阶段
    """
    # 步骤1：提取当前状态并检查早期退出条件
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # 早期退出：如果没有工具调用（包括原生网络搜索）
    # 这意味着AI已经完成思考，不需要进一步研究
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
            openai_websearch_called(most_recent_message) or
            anthropic_websearch_called(most_recent_message)
    )

    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")

    # 步骤2：处理其他工具调用（搜索、MCP工具等）
    # 获取工具映射，通过名称快速查找
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool
        for tool in tools
    }

    # 并行执行所有工具调用
    # 这提高了效率，特别是当需要多次搜索时
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # 从执行结果创建工具消息
    # 这些消息会返回给研究人员，用于下一轮决策
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # 步骤3：检查晚期退出条件（在处理工具后）
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        # 结束研究并进入压缩阶段
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    # 继续研究循环，包含工具结果
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """压缩并将研究结果综合成简洁、结构化的摘要。

    这是研究人员子图的最后一个节点，负责将所有研究发现压缩成摘要。

    为什么需要压缩？
    - 研究过程会产生大量消息和工具输出
    - 监督者需要简洁的摘要来理解研究结果
    - 避免token限制，保持上下文精简

    重试逻辑：
    - 如果遇到token限制，删除旧消息后重试
    - 最多重试3次
    - 如果仍失败，返回错误消息

    输出内容：
    - compressed_research: 压缩后的研究摘要
    - raw_notes: 原始笔记（未压缩的详细信息）

    Args:
        state: 当前研究人员状态，包含累积的研究消息
        config: 运行时配置，包含压缩模型设置

    Returns:
        包含压缩研究摘要和原始笔记的字典
    """
    # 步骤1：配置压缩模型
    # 使用单独的模型进行压缩，通常选择成本较低的模型
    configurable = Configuration.from_runnable_config(config)

    synthesizer_model = ((
        ChatOpenAI(model=configurable.compression_model,
                   max_tokens=configurable.compression_model_max_tokens,
                   api_key=get_api_key_for_model(configurable.compression_model, config),
                   base_url=get_base_url_for_model(configurable.compression_model),
                   tags=["langsmith:nostream"])
    ))

    # 步骤2：为压缩准备消息
    researcher_messages = state.get("researcher_messages", [])

    # 添加指令，从研究模式切换到压缩模式
    # 这告诉AI停止研究，开始总结
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    # 步骤3：尝试压缩，包含token限制的重试逻辑
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # 创建专注于压缩任务的系统提示
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            # 执行压缩
            response = await synthesizer_model.ainvoke(messages)

            # 从所有工具和AI消息中提取原始笔记
            # 这些是未经过滤的详细信息
            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            # 返回成功的压缩结果
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }

        except Exception as e:
            synthesis_attempts += 1

            # 通过删除旧消息处理token限制超出
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # 对于其他错误，继续重试
            continue

    # 步骤4：如果所有尝试失败，返回错误结果
    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """生成最终的综合研究报告，包含token限制的重试逻辑。

    这是整个研究流程的最后一个节点，将所有研究结果综合成最终报告。

    工作流程：
    1. 提取所有研究笔记和发现
    2. 配置报告生成模型
    3. 尝试生成报告，如果遇到token限制则截断内容后重试
    4. 返回最终报告

    渐进式截断策略：
    - 第一次重试：使用模型token限制的4倍作为字符限制
    - 后续重试：每次减少10%的内容
    - 这样可以在保持信息完整性的同时避免token限制

    Args:
        state: 包含研究结果和上下文的代理状态
        config: 包含模型设置和API密钥的运行时配置

    Returns:
        包含最终报告和已清理状态的字典
    """
    # 步骤1：提取研究结果并准备状态清理
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # 步骤2：配置最终报告生成模型
    configurable = Configuration.from_runnable_config(config)
    report_model = ((
        ChatOpenAI(model=configurable.final_report_model,
                   max_tokens=configurable.final_report_model_max_tokens,
                   api_key=get_api_key_for_model(configurable.final_report_model, config),
                   base_url=get_base_url_for_model(configurable.final_report_model),
                   tags=["langsmith:nostream"])
    ))

    # 步骤3：尝试报告生成，包含token限制的重试逻辑
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # 创建包含所有研究上下文的综合提示
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            # 生成最终报告
            final_report = await report_model.ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            # 返回成功的报告生成
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            # 使用渐进式截断处理token限制超出错误
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    # 第一次重试：确定初始截断限制
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # 使用4倍token限制作为字符近似截断
                    findings_token_limit = model_token_limit * 4
                else:
                    # 后续重试：每次减少10%
                    findings_token_limit = int(findings_token_limit * 0.9)

                # 截断发现内容并重试
                findings = findings[:findings_token_limit]
                continue
            else:
                # 非token限制错误：立即返回错误
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    # 步骤4：如果所有重试耗尽，返回失败结果
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


# ------------------------------------- 监督者子图构建 -------------------------------------
# 创建监督者工作流程，管理研究委派和协调
supervisor_builder = StateGraph(state_schema=SupervisorState, context_schema=Configuration)

# 添加监督者节点用于研究管理
supervisor_builder.add_node("supervisor", supervisor)
# 工具节点
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

# 定义监督者工作流程边
supervisor_builder.add_edge(START, "supervisor")  # 监督者入口点

# 编译监督者子图用于主工作流程中使用
supervisor_subgraph = supervisor_builder.compile()

# -------------------------------- 研究人员子图构建 --------------------------------
# 创建单个研究人员工作流程，对特定主题进行深入研究
researcher_builder = StateGraph(state_schema=ResearcherState, output_schema=ResearcherOutputState,
                                context_schema=Configuration)

# 添加研究人员节点用于研究执行和压缩
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)

# 定义研究人员工作流程边
researcher_builder.add_edge(START, "researcher")  # 研究人员入口点
researcher_builder.add_edge("compress_research", END)  # 压缩后退出点

# 编译研究人员子图供监督者并行执行
researcher_subgraph = researcher_builder.compile()

# 主深度研究图构建
# 创建从用户输入到最终报告的完整深度研究工作流程
deep_researcher_builder = StateGraph(
    state_schema=AgentState,
    input_schema=AgentInputState,
    context_schema=Configuration
)

# ---------------------------- 添加主工作流程节点，用于完整的研究过程 ----------------------------

# 用户澄清阶段
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
# 研究计划阶段
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
# 研究执行阶段
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
# 报告生成阶段
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 定义主工作流程边，用于顺序执行
deep_researcher_builder.add_edge(START, "clarify_with_user")  # 入口点
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")  # 研究到报告
deep_researcher_builder.add_edge("final_report_generation", END)  # 最终退出点

# 编译完整的深度研究工作流程
# 这是LangGraph的入口点，可以被外部调用
deep_researcher = deep_researcher_builder.compile()
