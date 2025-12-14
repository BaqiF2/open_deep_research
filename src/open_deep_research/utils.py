"""深度研究智能体的工具函数和辅助函数。"""

import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from string import Template
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts_zh import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

##########################
# Tavily 搜索工具工具函数
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "专为全面、准确和可信结果优化的搜索引擎。"
    "适用于需要回答当前事件相关问题的情况。"
)


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
        queries: List[str],
        max_results: Annotated[int, InjectedToolArg] = 5,
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
        config: RunnableConfig = None
) -> str:
    """从 Tavily 搜索 API 获取并总结搜索结果。

    Args:
        queries: 要执行的搜索查询列表
        max_results: 每个查询返回的最大结果数量
        topic: 搜索结果的主题过滤器（general、news 或 finance）
        config: API 密钥和模型设置的运行时配置

    Returns:
        包含总结搜索结果的格式化字符串
    """
    # 步骤 1: 异步执行搜索查询
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )

    # 步骤 2: 通过 URL 去重结果，避免多次处理相同内容
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}

    # 步骤 3: 使用配置设置总结模型
    configurable = Configuration.from_runnable_config(config)

    # 字符限制以保持在模型令牌限制内（可配置）
    max_char_to_include = configurable.max_content_length

    # 使用重试逻辑初始化总结模型
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    base_url = get_base_url_for_model(configurable.summarization_model)
    summarization_model = ChatOpenAI(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        base_url=base_url,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )

    # 步骤 4: 创建总结任务（跳过空内容）
    async def noop():
        """对于没有原始内容的结果的空操作函数。

        Returns:
            None: 始终返回 None。
        """
        return None

    summarization_tasks = [
        noop() if not result.get("raw_content")
        else summarize_webpage(
            summarization_model,
            result['raw_content'][:max_char_to_include]
        )
        for result in unique_results.values()
    ]

    # 步骤 5: 并行执行所有总结任务
    summaries = await asyncio.gather(*summarization_tasks)

    # 步骤 6: 将结果与其总结结合
    # 这段代码创建了一个字典推导式，将处理后的搜索结果组织成结构化格式
    # 
    # 数据结构说明：
    # - 外层键(url): 搜索结果的网页URL，确保每个URL只出现一次
    # - 内层字典包含两个字段：
    #   * 'title': 网页标题，直接从搜索结果中获取
    #   * 'content': 网页内容或总结
    #     - 如果 summary 为 None（表示总结失败或无原始内容），使用原始的 content
    #     - 否则使用 AI 生成的 summary（包含摘要和关键摘录）
    # 
    # zip() 函数说明：
    # - 将三个可迭代对象（URLs、结果字典、总结列表）并行迭代
    # - 确保 URL、对应的结果和对应的总结一一对应
    # - 例如：第1个URL对应第1个result和第1个summary
    summarized_results = {
        url: {
            'title': result['title'],
            'content': result['content'] if summary is None else summary
        }
        for url, result, summary in zip(
            unique_results.keys(),
            unique_results.values(),
            summaries
        )
    }

    # 步骤 7: 格式化最终输出
    if not summarized_results:
        return "未找到有效的搜索结果。请尝试不同的搜索查询或使用其他搜索API。"

    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i + 1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


# TODO 原子方法，制作搜索。考虑是否区分原子方法
async def tavily_search_async(
        search_queries,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = True,
        config: RunnableConfig = None
):
    """异步执行多个 Tavily 搜索查询。

    Args:
        search_queries: 要执行的搜索查询字符串列表
        max_results: 每个查询的最大结果数量
        topic: 用于过滤结果的主题类别
        include_raw_content: 是否包含完整的网页内容
        config: API 密钥访问的运行时配置

    Returns:
        来自 Tavily API 的搜索结果字典列表
    """
    # 使用配置中的 API 密钥初始化 Tavily 客户端
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))

    # 为并行执行创建搜索任务
    search_tasks = [
        tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        for query in search_queries
    ]

    # 并行执行所有搜索查询并返回结果
    search_results = await asyncio.gather(*search_tasks)
    return search_results


async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """使用 AI 模型总结网页内容，并提供超时保护。

    Args:
        model: 配置用于总结的聊天模型
        webpage_content: 要总结的原始网页内容

    Returns:
        带关键摘录的格式化总结，如果总结失败则返回原始内容
    """
    try:
        # 创建带有当前日期上下文的提示
        prompt_content = Template(summarize_webpage_prompt).substitute(
            webpage_content=webpage_content,
            date=get_today_str()
        )

        # asyncio.wait_for 是 Python 异步编程中用于设置超时的工具函数
        # 它接收两个主要参数：
        # 1. 一个协程对象（这里是 model.ainvoke 调用）
        # 2. timeout 超时时间（单位：秒）
        # 
        # 工作原理：
        # - 如果协程在指定时间内完成，返回协程的结果
        # - 如果超过指定时间还未完成，抛出 asyncio.TimeoutError 异常
        # - 这样可以防止某个异步操作无限期地等待，避免程序卡死
        #
        # 在这个场景中：
        # - model.ainvoke() 是调用 AI 模型进行网页内容总结的异步操作
        # - 设置 60 秒超时，如果 AI 模型 60 秒内没有返回结果，就会触发超时
        # - 超时后会被下面的 except asyncio.TimeoutError 捕获并返回原始内容
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0  # 总结的超时时间为 60 秒
        )

        # 使用结构化部分格式化总结
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except asyncio.TimeoutError:
        # 总结期间超时 - 返回原始内容
        logging.warning("总结在 60 秒后超时，返回原始内容")
        return webpage_content
    except Exception as e:
        # 总结期间的其他错误 - 记录日志并返回原始内容
        logging.warning(f"总结失败，错误: {str(e)}，返回原始内容")
        return webpage_content


##########################
# 反思工具工具函数
##########################
## FIXME 因为要调用工具 将提示词直接笑到工具的 docstring中，这使得模型需要去推理反思内容。类似todo-list的方式，很好的一个思路和参考
@tool(description="用于研究规划的战略反思工具")
def think_tool(reflection: str) -> str:
    """用于研究进度和决策的战略反思工具。

    在每次搜索后使用此工具分析结果并系统地规划下一步。
    这会在研究工作流中创建一个有意的暂停，以便进行高质量的决策。

    使用时机：
    - 收到搜索结果后：我发现了什么关键信息？
    - 决定下一步之前：我有足够的信息来全面回答吗？
    - 评估研究差距时：我仍然缺少什么具体信息？
    - 结束研究之前：我现在能提供完整的答案吗？

    反思应涵盖：
    1. 当前发现的分析 - 我收集了什么具体信息？
    2. 差距评估 - 仍然缺少什么关键信息？
    3. 质量评估 - 我有足够的证据/例子来提供好答案吗？
    4. 战略决策 - 我应该继续搜索还是提供答案？

    Args:
        reflection: 您对研究进度、发现、差距和下一步的详细反思

    Returns:
        确认反思已记录用于决策
    """
    return f"反思已记录：{reflection}"


##########################
# MCP 工具函数
##########################
# TODO MCP 访问方式有待商榷
async def get_mcp_access_token(
        supabase_token: str,
        base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """使用 OAuth 令牌交换将 Supabase 令牌交换为 MCP 访问令牌。

    Args:
        supabase_token: 有效的 Supabase 身份验证令牌
        base_mcp_url: MCP 服务器的基础 URL

    Returns:
        如果成功则返回令牌数据字典，失败则返回 None
    """
    try:
        # 准备 OAuth 令牌交换请求数据
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }

        # 执行令牌交换请求
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # 成功获取令牌
                    token_data = await response.json()
                    return token_data
                else:
                    # 记录错误详情用于调试
                    response_text = await response.text()
                    logging.error(f"令牌交换失败: {response_text}")

    except Exception as e:
        logging.error(f"令牌交换期间出错: {e}")

    return None


async def get_tokens(config: RunnableConfig):
    """检索存储的身份验证令牌并验证过期时间。

    Args:
        config: 包含线程和用户标识符的运行时配置

    Returns:
        如果有效且未过期则返回令牌字典，否则返回 None
    """
    store = get_store()

    # 从配置中提取必需的标识符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None

    # 检索存储的令牌
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None

    # 检查令牌过期时间
    expires_in = tokens.value.get("expires_in")  # 到期的秒数
    created_at = tokens.created_at  # 令牌创建的 datetime
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)

    if current_time > expiration_time:
        # 令牌已过期，清理并返回 None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value


async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """在配置存储中存储身份验证令牌。

    Args:
        config: 包含线程和用户标识符的运行时配置
        tokens: 要存储的令牌字典
    """
    store = get_store()

    # 从配置中提取必需的标识符
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return

    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return

    # 存储令牌
    await store.aput((user_id, "tokens"), "data", tokens)


async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """获取并刷新 MCP 令牌，必要时获取新的令牌。

    Args:
        config: 包含身份验证详情的运行时配置

    Returns:
        有效的令牌字典，如果无法获取令牌则返回 None
    """
    # 首先尝试获取现有的有效令牌
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens

    # 提取用于新令牌交换的 Supabase 令牌
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None

    # 提取 MCP 配置
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None

    # 将 Supabase 令牌交换为 MCP 令牌
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # 存储新令牌并返回它们
    await set_tokens(config, mcp_tokens)
    return mcp_tokens


def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """使用全面的身份验证和错误处理包装 MCP 工具。

    Args:
        tool: 要包装的 MCP 结构化工具

    Returns:
        具有身份验证错误处理的增强工具
    """
    original_coroutine = tool.coroutine

    async def authentication_wrapper(**kwargs):
        """具有 MCP 错误处理和用户友好消息的增强协程。

        Args:
            **kwargs: 传递给原始工具协程的关键字参数。

        Returns:
            原始工具协程的执行结果。

        Raises:
            ToolException: 当遇到需要交互的 MCP 错误时抛出。
            BaseException: 当遇到其他非 MCP 错误时重新抛出。
        """

        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """在异常链中递归搜索 MCP 错误。

            Args:
                exc: 要检查的异常对象。

            Returns:
                找到的 McpError 对象，如果未找到则返回 None。
            """
            if isinstance(exc, McpError):
                return exc

            # 通过检查属性处理 ExceptionGroup (Python 3.11+)
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None

        try:
            # 执行原始工具功能
            return await original_coroutine(**kwargs)

        except BaseException as original_error:
            # 在异常链中搜索 MCP 特定的错误
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # 不是 MCP 错误，重新抛出原始异常
                raise original_error

            # 处理 MCP 特定的错误情况
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}

            # 检查身份验证/交互所需的错误
            if error_code == -32003:  # 交互所需的错误代码
                message_payload = error_data.get("message", {})
                error_message = "需要交互"

                # 如果可用，提取用户友好的消息
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message

                # 如果提供了 URL，则追加以供用户参考
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"

                raise ToolException(error_message) from original_error

            # 对于其他 MCP 错误，重新抛出原始错误
            raise original_error

    # 将工具的协程替换为我们的增强版本
    tool.coroutine = authentication_wrapper
    return tool

##########################
# 工具工具函数
##########################
async def get_search_tool(search_api: SearchAPI):
    """基于指定的 API 提供程序配置并返回搜索工具。

    Args:
        search_api: 要使用的搜索 API 提供程序（Tavily 或 None）

    Returns:
        指定提供程序的已配置搜索工具对象列表
    """
    if search_api == SearchAPI.TAVILY:
        # 配置带有元数据的 Tavily 搜索工具
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search"
        }
        return [search_tool]

    elif search_api == SearchAPI.NONE:
        # 未配置搜索功能
        return []

    # 未知搜索 API 类型的默认回退
    return []


async def get_all_tools(config: RunnableConfig):
    """组装包含研究、思考、搜索。

    Args:
        config: 指定搜索 API 和 MCP 设置的运行时配置

    Returns:
        研究操作的所有已配置和可用工具列表
    """
    # 从核心研究工具开始
    tools = [ResearchComplete, think_tool]

    # 添加已配置的搜索工具
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """提取工具消息内容。

    Args:
        messages: 消息列表，包含各种类型的消息对象。

    Returns:
        工具消息内容的字符串列表。
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# 模型提供程序原生网络搜索工具函数
##########################

def anthropic_websearch_called(response):
    """检测响应中是否使用了 Anthropic 的原生网络搜索。

    Args:
        response: 来自 Anthropic API 的响应对象

    Returns:
        如果调用了网络搜索则返回 True，否则返回 False
    """
    try:
        # 浏览响应元数据结构
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # 检查服务器端工具使用信息
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # 查找网络搜索请求计数
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # 如果有任何网络搜索请求，返回 True
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # 处理响应结构意外的情况
        return False


def openai_websearch_called(response):
    """检测响应中是否使用了 OpenAI 的网络搜索功能。

    Args:
        response: 来自 OpenAI API 的响应对象

    Returns:
        如果调用了网络搜索则返回 True，否则返回 False
    """
    # 检查响应元数据中的工具输出
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # 在工具输出中查找网络搜索调用
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


##########################
# 令牌限制超限工具函数
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """确定异常是否表示令牌/上下文限制被超出。

    Args:
        exception: 要分析的异常
        model_name: 可选的模型名称，用于优化提供程序检测

    Returns:
        如果异常表示令牌限制被超出则返回 True，否则返回 False
    """
    error_str = str(exception).lower()

    # 步骤 1: 如果可用，从模型名称确定提供程序
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'

    # 步骤 2: 检查提供程序特定的令牌限制模式
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)

    # 步骤 3: 如果提供程序未知，检查所有提供程序
    return (
            _check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 OpenAI 令牌限制被超出。

    Args:
        exception: 要检查的异常对象。
        error_str: 异常消息的小写字符串表示。

    Returns:
        如果异常表示 OpenAI 令牌限制被超出则返回 True，否则返回 False。
    """
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 检查这是否是 OpenAI 异常
    is_openai_exception = (
            'openai' in exception_type.lower() or
            'openai' in module_name.lower()
    )

    # 检查典型的 OpenAI 令牌限制错误类型
    is_request_error = class_name in ['BadRequestError', 'InvalidRequestError']

    if is_openai_exception and is_request_error:
        # 在错误消息中查找令牌相关关键词
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # 检查特定的 OpenAI 错误代码
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        error_code = getattr(exception, 'code', '')
        error_type = getattr(exception, 'type', '')

        if (error_code == 'context_length_exceeded' or
                error_type == 'invalid_request_error'):
            return True

    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Anthropic 令牌限制被超出。

    Args:
        exception: 要检查的异常对象。
        error_str: 异常消息的小写字符串表示。

    Returns:
        如果异常表示 Anthropic 令牌限制被超出则返回 True，否则返回 False。
    """
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 检查这是否是 Anthropic 异常
    is_anthropic_exception = (
            'anthropic' in exception_type.lower() or
            'anthropic' in module_name.lower()
    )

    # 检查 Anthropic 特定的错误模式
    is_bad_request = class_name == 'BadRequestError'

    if is_anthropic_exception and is_bad_request:
        # Anthropic 对令牌限制使用特定的错误消息
        if 'prompt is too long' in error_str:
            return True

    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """检查异常是否表示 Google/Gemini 令牌限制被超出。

    Args:
        exception: 要检查的异常对象。
        error_str: 异常消息的小写字符串表示。

    Returns:
        如果异常表示 Gemini 令牌限制被超出则返回 True，否则返回 False。
    """
    # 分析异常元数据
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')

    # 检查这是否是 Google/Gemini 异常
    is_google_exception = (
            'google' in exception_type.lower() or
            'google' in module_name.lower()
    )

    # 检查 Google 特定的资源耗尽错误
    is_resource_exhausted = class_name in [
        'ResourceExhausted',
        'GoogleGenerativeAIFetchError'
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # 检查特定的 Google API 资源耗尽模式
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True

    return False


# 注意：这可能已过时或不适用于您的模型。请根据需要更新此内容。
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}


def get_model_token_limit(model_string):
    """查找特定模型的令牌限制。

    Args:
        model_string: 要查找的模型标识符字符串

    Returns:
        如果找到则返回整数形式的令牌限制，如果模型不在查找表中则返回 None
    """
    # 搜索已知的模型令牌限制
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit

    # 在查找表中未找到模型
    return None


def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """通过删除到最后一个 AI 消息为止来截断消息历史。

    这对于通过删除最近的上下文来处理令牌限制超限错误很有用。

    Args:
        messages: 要截断的消息对象列表

    Returns:
        截断的消息列表，到（但不包括）最后一个 AI 消息为止
    """
    # 向后搜索消息以找到最后一个 AI 消息
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # 返回到（但不包括）最后一个 AI 消息的所有内容
            return messages[:i]

    # 未找到 AI 消息，返回原始列表
    return messages


##########################
# 杂项工具函数
##########################
# TODO 非模型调用工具可与模型调用工具区分
def get_today_str() -> str:
    """获取当前日期并格式化以在提示和输出中显示。

    Returns:
        人类可读的日期字符串，格式如 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_config_value(value):
    """从配置中提取值，处理枚举和 None 值。

    Args:
        value: 要提取的配置值，可以是 None、字符串、字典或枚举。

    Returns:
        提取后的值。如果是枚举则返回其 value 属性，否则直接返回。
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """从环境或配置中获取特定模型的 API 密钥。

    Args:
        model_name: 模型名称，用于确定使用哪个 API 密钥。
        config: 运行时配置，可能包含 API 密钥信息。

    Returns:
        对应模型的 API 密钥字符串，如果未找到则返回 None。
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    # 这个基本不使用
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        else:
            return os.getenv("OPENAI_API_KEY")
    else:
        if model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        # 默认使用 OpenAIKEY
        else:
            return os.getenv("OPENAI_API_KEY")


def get_base_url_for_model(model_name: str):
    """从环境或配置中获取特定模型的 URL。

    Args:
        model_name: 模型名称，用于确定是否需要自定义 base URL。

    Returns:
        模型的 base URL 字符串，如果使用默认 URL 则返回 None。
    """
    model_name = model_name.lower()
    if model_name.startswith("openai:") or model_name.startswith("anthropic:") or model_name.startswith("google"):
        return None
    else:
        return os.getenv("BASE_URL")


def get_tavily_api_key(config: RunnableConfig):
    """从环境或配置中获取 Tavily API 密钥。

    Args:
        config: 运行时配置，可能包含 API 密钥信息。

    Returns:
        Tavily API 密钥字符串，如果未找到则返回 None。
    """
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")
