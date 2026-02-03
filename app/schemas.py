"""
 Модуль: schemas.py
 Назначение: Схемы данных (Pydantic-модели) для API
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════════════════════

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ToolName(str, Enum):
    CREATE_EVENT = "create_event"
    UPDATE_EVENT = "update_event"
    FIND_FREE_SLOT = "find_free_slot"
    SPLIT_TASK = "split_task"
    SUMMARIZE_WEEK = "summarize_week"
    GENERAL_CHAT = "general_chat"
    CLARIFICATION_NEEDED = "clarification_needed"

class DataRequirementType(str, Enum):
    NONE = "none"
    SLOTS = "slots"
    EVENTS = "events"


# ═══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЕ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

class UserContext(BaseModel):
    current_time: str = Field(..., description="ISO 8601")
    timezone: str

    work_start_hour: int = Field(default=9, description="Начало рабочего дня (0-23)")
    work_end_hour: int = Field(default=18, description="Конец рабочего дня (0-23)")


class CandidateSlot(BaseModel):
    start: str
    end: str


class ExistingEvent(BaseModel):
    id: Optional[str] = None
    title: str
    start: str
    end: str


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    text: str
    context: UserContext


class SearchConfig(BaseModel):
    start: str
    end: str
    min_duration_minutes: int = 30


class ExecuteRequest(BaseModel):
    text: str
    context: UserContext
    tool_name: ToolName
    fetched_slots: Optional[List[CandidateSlot]] = None
    fetched_events: Optional[List[ExistingEvent]] = None


class CreateEventParams(BaseModel):
    title: str
    start_time: str
    duration_minutes: int = 60

class RankedSlotParams(BaseModel):
    ranked_slots: List[CandidateSlot]
    reasoning: str

class SubTask(BaseModel):
    title: str
    duration_minutes: int

class SplitTaskParams(BaseModel):
    main_task: str
    subtasks: List[SubTask]

class MLResponse(BaseModel):
    tool_name: ToolName
    reply_text: str
    parameters: Union[RankedSlotParams, CreateEventParams, SplitTaskParams, Dict[str, Any]] = Field(default_factory=dict)

class AnalyzeResponse(BaseModel):
    tool_name: ToolName
    requirement: DataRequirementType
    data_params: Optional[SearchConfig] = None
    final_response: Optional[MLResponse] = None


class FeedbackRequest(BaseModel):
    context: UserContext
    chosen_slot: CandidateSlot
    rejected_slots: List[CandidateSlot]