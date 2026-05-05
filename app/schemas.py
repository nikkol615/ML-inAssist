"""
 Модуль: schemas.py
 Назначение: Схемы данных (Pydantic-модели) для API
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


def _parse_iso_datetime(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty ISO 8601 datetime string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_iso_datetime(value: Optional[str]) -> Optional[str]:
    if value is not None:
        _parse_iso_datetime(value)
    return value


def _validate_datetime_order(start: str, end: str) -> None:
    start_dt = _parse_iso_datetime(start)
    end_dt = _parse_iso_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("end must be later than start")


class ToolName(str, Enum):
    CREATE_EVENT = "create_event"
    UPDATE_EVENT = "update_event"
    DELETE_EVENT = "delete_event"
    FIND_FREE_SLOT = "find_free_slot"
    SPLIT_TASK = "split_task"
    SUMMARIZE_WEEK = "summarize_week"
    GENERAL_CHAT = "general_chat"
    CLARIFICATION_NEEDED = "clarification_needed"


class DataRequirementType(str, Enum):
    NONE = "none"
    SLOTS = "slots"
    EVENTS = "events"


class GatewayToolName(str, Enum):
    FIND_EVENT = "find_event"
    LIST_EVENTS = "list_events"
    GET_FREE_SLOTS = "get_free_slots"
    CREATE_EVENT = "create_event"
    UPDATE_EVENT = "update_event"
    DELETE_EVENT = "delete_event"


class GatewayActionType(str, Enum):
    NONE = "none"
    TOOL_CALL = "tool_call"


class AgentResponseType(str, Enum):
    TOOL_CALL = "tool_call"
    CLARIFY = "clarify"
    FINISH = "finish"


class ActionStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class UserContext(BaseModel):
    current_time: str = Field(..., description="ISO 8601")
    timezone: str = Field(..., min_length=1, max_length=64)
    work_start_hour: int = Field(default=9, ge=0, le=23, description="Начало рабочего дня (0-23)")
    work_end_hour: int = Field(default=18, ge=1, le=24, description="Конец рабочего дня (1-24)")

    @field_validator("current_time")
    @classmethod
    def validate_current_time(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def validate_work_hours(self):
        if self.work_end_hour <= self.work_start_hour:
            raise ValueError("work_end_hour must be greater than work_start_hour")
        return self


class CandidateSlot(BaseModel):
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def validate_slot_datetime(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def validate_slot_order(self):
        _validate_datetime_order(self.start, self.end)
        return self


class ExistingEvent(BaseModel):
    id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    title: str = Field(..., min_length=1, max_length=512)
    start: str
    end: str
    location: Optional[str] = Field(default=None, max_length=512)
    description: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("start", "end")
    @classmethod
    def validate_event_datetime(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def validate_event_order(self):
        _validate_datetime_order(self.start, self.end)
        return self


class ConversationMessage(BaseModel):
    role: str = Field(..., min_length=1, max_length=32)
    text: str = Field(..., min_length=1, max_length=10000)


class PendingAction(BaseModel):
    tool_name: Optional[Union[ToolName, GatewayToolName, str]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    stage: Optional[str] = None
    status: Optional[str] = None


class ConversationContext(BaseModel):
    last_user_message: Optional[str] = None
    last_assistant_message: Optional[str] = None
    pending_action: Optional[PendingAction] = None
    recent_messages: List[ConversationMessage] = Field(default_factory=list)


class GatewayToolCall(BaseModel):
    tool_name: GatewayToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)


class GatewayAction(BaseModel):
    type: GatewayActionType = GatewayActionType.NONE
    tool_call: Optional[GatewayToolCall] = None


class CompletedAction(BaseModel):
    tool_name: GatewayToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.SUCCESS
    summary: Optional[str] = None


class ToolObservation(BaseModel):
    tool_name: GatewayToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.SUCCESS
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class WorkingState(BaseModel):
    goal: Optional[str] = None
    status: str = "idle"
    plan_summary: Optional[str] = None
    pending_action: Dict[str, Any] = Field(default_factory=dict)
    resolved_entities: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    messages: List[ConversationMessage] = Field(default_factory=list)
    completed_actions: List[CompletedAction] = Field(default_factory=list)
    tool_observations: List[ToolObservation] = Field(default_factory=list)
    working_state: WorkingState = Field(default_factory=WorkingState)
    memory_summary: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    context: UserContext
    conversation: Optional[ConversationContext] = None
    all_context: Optional[str] = Field(default=None, max_length=50000)


class SearchConfig(BaseModel):
    start: str
    end: str
    min_duration_minutes: int = Field(default=30, ge=1, le=1440)

    @field_validator("start", "end")
    @classmethod
    def validate_search_datetime(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def validate_search_order(self):
        _validate_datetime_order(self.start, self.end)
        return self


class ExecuteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    context: UserContext
    tool_name: ToolName
    fetched_slots: Optional[List[CandidateSlot]] = None
    fetched_events: Optional[List[ExistingEvent]] = None
    conversation: Optional[ConversationContext] = None
    all_context: Optional[str] = Field(default=None, max_length=50000)


class CreateEventParams(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    start_time: str
    duration_minutes: int = Field(default=60, ge=1, le=1440)
    location: Optional[str] = Field(default=None, max_length=512)
    description: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("start_time")
    @classmethod
    def validate_start_time(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value


class EventUpdates(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=512)
    start_time: Optional[str] = None
    duration_minutes: Optional[int] = Field(default=None, ge=1, le=1440)
    location: Optional[str] = Field(default=None, max_length=512)
    description: Optional[str] = Field(default=None, max_length=4000)

    @field_validator("start_time")
    @classmethod
    def validate_optional_start_time(cls, value: Optional[str]) -> Optional[str]:
        return _validate_iso_datetime(value)


class UpdateEventParams(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=256)
    updates: EventUpdates


class DeleteEventParams(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=256)


class RankedSlotParams(BaseModel):
    ranked_slots: List[CandidateSlot]
    reasoning: str = Field(..., min_length=1, max_length=1000)


class SubTask(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)
    duration_minutes: int = Field(..., ge=1, le=1440)


class SplitTaskParams(BaseModel):
    main_task: str = Field(..., min_length=1, max_length=512)
    subtasks: List[SubTask]


class MLResponse(BaseModel):
    tool_name: ToolName
    reply_text: str
    parameters: Union[
        RankedSlotParams,
        CreateEventParams,
        UpdateEventParams,
        DeleteEventParams,
        SplitTaskParams,
        Dict[str, Any],
    ] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    tool_name: ToolName
    requirement: DataRequirementType
    data_params: Optional[SearchConfig] = None
    final_response: Optional[MLResponse] = None


class StepRequest(BaseModel):
    text: str = Field(default="", max_length=10000)
    context: UserContext
    state: Optional[AgentState] = None
    conversation: Optional[ConversationContext] = None
    all_context: Optional[str] = Field(default=None, max_length=50000)


class StepResponse(BaseModel):
    response_type: AgentResponseType
    assistant_message: Optional[str] = None
    response_payload: Dict[str, Any] = Field(default_factory=dict)
    next_gateway_action: GatewayAction = Field(default_factory=GatewayAction)
    state_patch: Dict[str, Any] = Field(default_factory=dict)


class FindEventToolArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    max_results: int = Field(default=10, ge=1, le=50)

    @field_validator("time_min", "time_max")
    @classmethod
    def validate_find_time(cls, value: Optional[str]) -> Optional[str]:
        return _validate_iso_datetime(value)

    @model_validator(mode="after")
    def validate_find_time_order(self):
        if self.time_min and self.time_max:
            _validate_datetime_order(self.time_min, self.time_max)
        return self


class ListEventsToolArgs(BaseModel):
    start: str
    end: str
    query: Optional[str] = Field(default=None, min_length=1, max_length=512)
    max_results: int = Field(default=20, ge=1, le=100)

    @field_validator("start", "end")
    @classmethod
    def validate_list_datetime(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def validate_list_order(self):
        _validate_datetime_order(self.start, self.end)
        return self


class GetFreeSlotsToolArgs(BaseModel):
    start: str
    end: str
    min_duration_minutes: int = Field(default=30, ge=1, le=1440)

    @field_validator("start", "end")
    @classmethod
    def validate_free_slot_datetime(cls, value: str) -> str:
        _validate_iso_datetime(value)
        return value

    @model_validator(mode="after")
    def validate_free_slot_order(self):
        _validate_datetime_order(self.start, self.end)
        return self


class FeedbackRequest(BaseModel):
    context: UserContext
    chosen_slot: CandidateSlot
    rejected_slots: List[CandidateSlot]
