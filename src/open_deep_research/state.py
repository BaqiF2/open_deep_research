"""深度研究智能体的图状态定义和数据结构。"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# 结构化输出
###################
class ConductResearch(BaseModel):
    """调用此工具对特定主题进行研究。

    Attributes:
        research_topic: 要研究的主题。应该是单个主题，并且应该详细描述（至少一段）。
    """

    research_topic: str = Field(
        description="要研究的主题。应该是单个主题，并且应该详细描述（至少一段）。",
    )


class ResearchComplete(BaseModel):
    """调用此工具表示研究完成。"""


class Summary(BaseModel):
    """研究摘要及关键发现。

    Attributes:
        summary: 研究内容的摘要。
        key_excerpts: 关键摘录和引用。
    """

    summary: str
    key_excerpts: str


class ClarifyWithUser(BaseModel):
    """用户澄清请求的模型。

    Attributes:
        need_clarification: 是否需要向用户询问澄清问题。
        question: 向用户询问以澄清报告范围的问题。
        verification: 验证消息，表示在用户提供必要信息后将开始研究。
    """

    need_clarification: bool = Field(
        description="是否需要向用户询问澄清问题。",
    )
    question: str = Field(
        description="向用户询问以澄清报告范围的问题",
    )
    verification: str = Field(
        description="验证消息，表示在用户提供必要信息后将开始研究。",
    )


class ResearchQuestion(BaseModel):
    """用于指导研究的研究问题和简要说明。

    Attributes:
        research_brief: 将用于指导研究的研究问题。
    """

    research_brief: str = Field(
        description="将用于指导研究的研究问题。",
    )


###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """允许覆盖状态中值的Reducer函数。

    Args:
        current_value: 当前状态中的值。
        new_value: 要应用的新值。如果是包含 type='override' 的字典，则完全替换；
            否则使用 operator.add 进行累加。

    Returns:
        更新后的值。
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """InputState仅包含'messages'。

    继承自 MessagesState，用于定义代理的输入状态。
    """


class AgentState(MessagesState):
    """包含消息和研究数据的主要智能体状态。

    Attributes:
        supervisor_messages: 存储与研究相关的消息。
        research_brief: 存储当前研究主题。
        raw_notes: 存储原始笔记。
        notes: 存储笔记。
        final_report: 存储最终报告。
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    final_report: str


class SupervisorState(TypedDict):
    """管理研究任务的监督者状态。

    Attributes:
        supervisor_messages: 存储与研究相关的消息。
        research_brief: 存储当前研究主题。
        notes: 存储笔记。
        research_iterations: 存储研究迭代次数。
        raw_notes: 存储原始笔记。
    """

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int
    raw_notes: Annotated[list[str], override_reducer]


class ResearcherState(TypedDict):
    """研究者状态。

    Attributes:
        researcher_messages: 存储与研究相关的消息。
        tool_call_iterations: 工具调用次数。
        research_topic: 研究主题。
        compressed_research: 压缩研究结果。
        raw_notes: 存储原始笔记。
    """

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]


class ResearcherOutputState(BaseModel):
    """研究者的输出状态。

    Attributes:
        compressed_research: 存储压缩后的研究结果。
        raw_notes: 存储原始笔记。
    """

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
