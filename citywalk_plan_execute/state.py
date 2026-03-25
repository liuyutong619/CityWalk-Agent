"""State definitions for CityWalk Plan and Execute agent."""

import operator
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rag.schema import PlannerNoteContext


###################
# Structured Outputs
###################

class UserIntent(BaseModel):
    """Parsed user intent from query."""

    city: str = Field(description="城市名称")
    start_location: Optional[str] = Field(default=None, description="起点位置")
    end_location: Optional[str] = Field(default=None, description="终点位置")
    return_to_start: bool = Field(default=False, description="是否需要最终回到起点")
    activity_type: str = Field(description="活动类型：步行/骑车/打卡等")
    preferences: list[str] = Field(default_factory=list, description="用户偏好列表")
    constraints: dict = Field(default_factory=dict, description="约束条件")


class IntentClarification(BaseModel):
    """Intent clarification result."""

    is_clear: bool = Field(description="意图是否清晰")
    missing_info: list[str] = Field(default_factory=list, description="缺失的信息")
    clarification_question: Optional[str] = Field(default=None, description="澄清问题")


class EvaluationResult(BaseModel):
    """Evaluation of execution results."""

    satisfied: bool = Field(description="是否满足用户需求")
    issues: list[str] = Field(default_factory=list, description="发现的问题")
    suggestions: list[str] = Field(default_factory=list, description="改进建议")


###################
# Supervisor Dispatch Tools (Pydantic models for bind_tools)
###################

class DispatchPOIExplorer(BaseModel):
    """调用此工具派遣 POI Explorer 去探索兴趣点。"""
    task_description: str = Field(description="给 POI Explorer 的具体探索任务描述")


class DispatchRoutePlanner(BaseModel):
    """调用此工具派遣 Route Planner 去规划路线。"""
    task_description: str = Field(description="给 Route Planner 的具体规划任务描述")


class AllTasksComplete(BaseModel):
    """所有子任务已完成，准备输出最终结果。"""
    summary: str = Field(description="整体完成情况总结")


###################
# Sub-Agent States
###################

class POIExplorerState(TypedDict):
    """POI Explorer 子图内部状态"""
    messages: Annotated[list, operator.add]  # ReAct 对话历史
    tool_call_iterations: int
    _candidate_pool: list[dict]  # 原始搜索结果暂存（未经筛选）
    explored_pois: list[dict]   # LLM 精选后的 POI（ExplorationComplete 时写入）


class RoutePlannerState(TypedDict):
    """Route Planner 子图内部状态"""
    messages: Annotated[list, operator.add]
    tool_call_iterations: int
    route_records: list[dict]  # 本轮规划的路线记录


class RouteNearbyPOIState(TypedDict):
    """路线周边可选点补充子图内部状态"""
    messages: Annotated[list, operator.add]
    tool_call_iterations: int
    route_polyline: str
    selected_stops: list[dict]
    explored_pois: list[dict]
    _candidate_pool: list[dict]
    nearby_route_pois: list[dict]


###################
# Main Agent State
###################

class AgentState(TypedDict):
    """Main agent state."""

    # 用户输入（支持多轮对话）
    user_query: str
    conversation_history: Annotated[list[dict], operator.add]

    # 意图澄清结果
    clarification: Optional[IntentClarification]

    # 解析后的意图
    intent: Optional[UserIntent]

    # Supervisor 对话历史 (AIMessage + ToolMessage 循环)
    supervisor_messages: Annotated[list, operator.add]

    # 各 sub-agent 已验收的成果
    retrieved_info: list[str]
    retrieved_note_contexts: list[PlannerNoteContext]
    explored_pois: list[dict]
    planned_routes: list[dict]
    route_plan: Optional[dict]
    nearby_route_pois: list[dict]

    # 执行结果 (parse_intent 仍需写入坐标结果)
    execution_results: list[dict]

    # 控制
    supervisor_iterations: int
    replan_count: int

    # 评估结果
    evaluation: Optional[EvaluationResult]

    # 最终输出
    final_output: Optional[dict]
