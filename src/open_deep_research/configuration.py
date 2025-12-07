"""Open Deep Research 系统的配置管理。"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """可用的搜索 API 提供者枚举。"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """模型上下文协议 (MCP) 服务器的配置。"""

    url: Optional[str] = Field(
        default=None,
    )
    """MCP 服务器的 URL"""
    tools: Optional[List[str]] = Field(
        default=None,
    )
    """提供给 LLM 的工具列表"""
    auth_required: Optional[bool] = Field(
        default=False,
    )
    """MCP 服务器是否需要认证"""

class Configuration(BaseModel):
    """深度研究代理的主要配置类。"""

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
    # 模型配置
    summarization_model: str = Field(
        default="openai:gpt-4.1-mini",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
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
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
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
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
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
        default="openai:gpt-4.1",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
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
        """从 RunnableConfig 创建 Configuration 实例。"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic 配置。"""

        arbitrary_types_allowed = True