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
    """调用此工具对特定主题进行研究。"""
    research_topic: str = Field(
        description="要研究的主题。应该是单个主题，并且应该详细描述（至少一段）。",
    )

class ResearchComplete(BaseModel):
    """调用此工具表示研究完成。"""

class Summary(BaseModel):
    """研究摘要及关键发现。"""

    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """用户澄清请求的模型。"""

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
    """用于指导研究的研究问题和简要说明。"""

    research_brief: str = Field(
        description="将用于指导研究的研究问题。",
    )


###################
# 状态定义
###################

def override_reducer(current_value, new_value):
    """允许覆盖状态中值的Reducer函数。"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class AgentInputState(MessagesState):
    """InputState仅包含'messages'。"""

class AgentState(MessagesState):
    """包含消息和研究数据的主要智能体状态。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """管理研究任务的监督者状态。"""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """进行研究的个别研究者状态。"""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """来自个别研究者的输出状态。"""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []