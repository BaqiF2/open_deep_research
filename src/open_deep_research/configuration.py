"""Open Deep Research 系统的配置管理。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用的搜索 API 提供者枚举。

    Attributes:
        ANTHROPIC: Anthropic 搜索 API。
        OPENAI: OpenAI 搜索 API。
        TAVILY: Tavily 搜索 API。
        NONE: 不使用搜索 API。
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


class MCPConfig(BaseModel):
    """模型上下文协议 (MCP) 服务器的配置。

    Attributes:
        url: MCP 服务器的 URL。
        tools: 提供给 LLM 的工具列表。
        auth_required: MCP 服务器是否需要认证。
    """

    url: Optional[str] = Field(
        default=None,
    )
    tools: Optional[List[str]] = Field(
        default=None,
    )
    auth_required: Optional[bool] = Field(
        default=False,
    )


class Configuration(BaseModel):
    """深度研究代理的主要配置类。

    Attributes:
        max_structured_output_retries: 模型结构化输出调用的最大重试次数。
        allow_clarification: 是否允许研究者在开始研究前向用户询问澄清问题。
        max_concurrent_research_units: 并发运行的研究单元最大数量。
        search_api: 用于研究的搜索 API。
        max_researcher_iterations: 研究监督者的最大研究迭代次数。
        max_react_tool_calls: 单个研究者步骤中工具调用迭代的最大次数。
        summarization_model: 用于总结搜索结果的模型。
        summarization_model_max_tokens: 总结模型的最大输出令牌数。
        max_content_length: 总结前网页内容的最大字符长度。
        research_model: 用于进行研究的模型。
        research_model_max_tokens: 研究模型的最大输出令牌数。
        compression_model: 用于压缩子代理研究发现的模型。
        compression_model_max_tokens: 压缩模型的最大输出令牌数。
        final_report_model: 用于撰写最终报告的模型。
        final_report_model_max_tokens: 最终报告模型的最大输出令牌数。
        mcp_config: MCP 服务器配置。
        mcp_prompt: 关于可用 MCP 工具的额外指令。
    """

    # 通用配置
    max_structured_output_retries: int = Field(
        default=3,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "模型结构化输出调用的最大重试次数"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否允许研究者在开始研究前向用户询问澄清问题"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "并发运行的研究单元最大数量。这将允许研究者使用多个子代理进行研究。注意：并发数越高，可能会遇到速率限制。"
            }
        }
    )
    # 研究配置
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "用于研究的搜索 API。注意：确保你的研究者模型支持所选的搜索 API。",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI 原生网页搜索", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic 原生网页搜索", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "无", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "研究监督者的最大研究迭代次数。这是研究监督者反思研究并提出后续问题的次数。"
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "单个研究者步骤中工具调用迭代的最大次数。"
            }
        }
    )
    # 总结模型配置 TODO: 替换为 Qwen系列
    summarization_model: str = Field(
        default="qwen3-max",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "qwen3-max",
                "description": "用于总结 Tavily 搜索结果的研究结果的模型"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "总结模型的最大输出令牌数"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "总结前网页内容的最大字符长度"
            }
        }
    )
    research_model: str = Field(
        default="qwen3-max",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "qwen3-max",
                "description": "用于进行研究的模型。注意：确保你的研究者模型支持所选的搜索 API。"
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "研究模型的最大输出令牌数"
            }
        }
    )
    compression_model: str = Field(
        default="qwen3-max",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "qwen3-max",
                "description": "用于压缩子代理研究发现的模型。注意：确保你的压缩模型支持所选的搜索 API。"
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "压缩模型的最大输出令牌数"
            }
        }
    )
    final_report_model: str = Field(
        default="qwen3-max",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "qwen3-max",
                "description": "用于根据所有研究发现撰写最终报告的模型"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "最终报告模型的最大输出令牌数"
            }
        }
    )
    # MCP 服务器配置
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP 服务器配置"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "description": "传递给代理的关于可用 MCP 工具的任何额外指令。"
            }
        }
    )

    @classmethod
    def from_runnable_config(
            cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 创建 Configuration 实例。

        Args:
            config: 可选的 RunnableConfig 对象（LangChain 运行时配置）。
                如果提供，将从其 'configurable' 字典中提取配置值。
                配置优先级：环境变量 > configurable 字典 > 默认值。

        Returns:
            Configuration: 新创建的 Configuration 实例。

        Examples:
            >>> config = Configuration.from_runnable_config()
            >>> runnable_config = {"configurable": {"search_api": "tavily"}}
            >>> config = Configuration.from_runnable_config(runnable_config)
        """
        # 如果传入了 config，提取其中的 "configurable" 字典
        configurable = config.get("configurable", {}) if config else {}
        # 使用 Pydantic 的 model_fields 获取 Configuration 类定义的所有字段名
        field_names = list(cls.model_fields.keys())

        # 环境变量优先：os.environ.get(field_name.upper(), ...)
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        # 过滤掉值为 None 的键：只传递有实际值的配置,未传递的字段将使用
        # Pydantic 定义的默认值（如 max_structured_output_retries=3
        return cls(**{k: v for k, v in values.items() if v is not None})
