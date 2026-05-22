"""
Agent-step orchestration layer built on top of the legacy LLM service.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import app.prompts as p
from app.schemas import (
    ActionStatus,
    AnalyzeRequest,
    AnalyzeResponse,
    DataRequirementType,
    AgentResponseType,
    AgentState,
    CandidateSlot,
    ConversationContext,
    ConversationMessage,
    CreateEventParams,
    DeleteEventParams,
    ExecuteRequest,
    FindEventToolArgs,
    GatewayAction,
    GatewayActionType,
    GatewayToolCall,
    GatewayToolName,
    GetFreeSlotsToolArgs,
    ListEventsToolArgs,
    MLResponse,
    RankedSlotParams,
    SearchConfig,
    SplitTaskParams,
    StepRequest,
    StepResponse,
    ToolName,
    ToolObservation,
    UpdateEventParams,
)

logger = logging.getLogger("ml_agent")
logger.setLevel(logging.INFO)


class AgentService:
    """Primary runtime for the new /step architecture."""

    def __init__(self, llm_service):
        self.llm = llm_service

    def _tool_name_to_string(self, tool_name: Any) -> Optional[str]:
        if tool_name is None:
            return None
        return getattr(tool_name, "value", str(tool_name))

    def _build_synthetic_messages(
            self,
            conversation: Optional[ConversationContext],
    ) -> List[ConversationMessage]:
        if not conversation:
            return []
        if conversation.recent_messages:
            return list(conversation.recent_messages)

        messages: List[ConversationMessage] = []
        if conversation.last_user_message:
            messages.append(ConversationMessage(role="user", text=conversation.last_user_message))
        if conversation.last_assistant_message:
            messages.append(ConversationMessage(role="assistant", text=conversation.last_assistant_message))
        return messages

    def _conversation_to_state(self, conversation: Optional[ConversationContext]) -> AgentState:
        state = AgentState()
        if not conversation:
            return state

        state.messages = self._build_synthetic_messages(conversation)

        if conversation.pending_action:
            state.working_state.pending_action = conversation.pending_action.model_dump(
                mode="json",
                exclude_none=True,
            )
            state.working_state.status = conversation.pending_action.status or "in_progress"
            pending_tool_name = self._tool_name_to_string(conversation.pending_action.tool_name)
            if pending_tool_name:
                state.working_state.goal = pending_tool_name

        return state

    def _normalize_state(self, request: StepRequest) -> AgentState:
        if request.state is not None:
            state = request.state.model_copy(deep=True)
        else:
            state = self._conversation_to_state(request.conversation)

        if request.conversation:
            if not state.messages:
                state.messages = self._build_synthetic_messages(request.conversation)

            if not state.working_state.pending_action and request.conversation.pending_action:
                state.working_state.pending_action = request.conversation.pending_action.model_dump(
                    mode="json",
                    exclude_none=True
                )
                state.working_state.status = request.conversation.pending_action.status or state.working_state.status

            if not state.working_state.goal and request.conversation.pending_action:
                pending_tool_name = self._tool_name_to_string(request.conversation.pending_action.tool_name)
                if pending_tool_name:
                    state.working_state.goal = pending_tool_name

        return state

    def _prepare_state_for_prompt(self, state: AgentState, context) -> Dict[str, Any]:
        state_data = state.model_dump(mode="json", exclude_none=True)
        prepared_observations: List[Dict[str, Any]] = []

        for observation in state.tool_observations:
            obs_data = observation.model_dump(mode="json", exclude_none=True)
            if observation.tool_name == GatewayToolName.GET_FREE_SLOTS:
                result = dict(obs_data.get("result") or {})
                raw_slots = result.get("slots", [])
                valid_slots: List[CandidateSlot] = []
                if isinstance(raw_slots, list):
                    for item in raw_slots:
                        if not isinstance(item, dict):
                            continue
                        try:
                            valid_slots.append(CandidateSlot(**item))
                        except Exception:
                            continue
                if valid_slots:
                    ranked_slots = self.llm.ranker.rank_slots(valid_slots, context)
                    result["ranked_slots"] = [slot.model_dump(mode="json") for slot in ranked_slots[:3]]
                    obs_data["result"] = result
            prepared_observations.append(obs_data)

        state_data["tool_observations"] = prepared_observations
        return state_data

    def _build_step_payload(self, request: StepRequest, state_data: Dict[str, Any]) -> str:
        payload = {
            "text": request.text,
            "context": request.context.model_dump(mode="json"),
            "state": state_data,
        }
        if request.all_context:
            payload["all_context"] = request.all_context
        return json.dumps(payload, ensure_ascii=False)

    def _validate_tool_call(self, tool_call: GatewayToolCall) -> GatewayToolCall:
        raw_args = tool_call.arguments or {}

        if tool_call.tool_name == GatewayToolName.FIND_EVENT:
            args = FindEventToolArgs(**raw_args)
            if args.time_min and not self.llm._validate_iso_date(args.time_min):
                raise ValueError("Invalid find_event time_min")
            if args.time_max and not self.llm._validate_iso_date(args.time_max):
                raise ValueError("Invalid find_event time_max")
            return GatewayToolCall(
                tool_name=tool_call.tool_name,
                arguments=args.model_dump(mode="json", exclude_none=True),
            )

        if tool_call.tool_name == GatewayToolName.LIST_EVENTS:
            args = ListEventsToolArgs(**raw_args)
            if not self.llm._validate_iso_date(args.start) or not self.llm._validate_iso_date(args.end):
                raise ValueError("Invalid list_events date range")
            return GatewayToolCall(
                tool_name=tool_call.tool_name,
                arguments=args.model_dump(mode="json", exclude_none=True),
            )

        if tool_call.tool_name == GatewayToolName.GET_FREE_SLOTS:
            args = GetFreeSlotsToolArgs(**raw_args)
            if not self.llm._validate_iso_date(args.start) or not self.llm._validate_iso_date(args.end):
                raise ValueError("Invalid get_free_slots date range")
            return GatewayToolCall(
                tool_name=tool_call.tool_name,
                arguments=args.model_dump(mode="json", exclude_none=True),
            )

        if tool_call.tool_name == GatewayToolName.CREATE_EVENT:
            args = CreateEventParams(**raw_args)
            if not self.llm._validate_iso_date(args.start_time):
                raise ValueError("Invalid create_event start_time")
            return GatewayToolCall(
                tool_name=tool_call.tool_name,
                arguments=args.model_dump(mode="json", exclude_none=True),
            )

        if tool_call.tool_name == GatewayToolName.UPDATE_EVENT:
            args = UpdateEventParams(**raw_args)
            updates = args.updates
            if not any([
                updates.title,
                updates.start_time,
                updates.duration_minutes is not None,
                updates.location,
                updates.description,
            ]):
                raise ValueError("update_event requires at least one update")
            if updates.start_time and not self.llm._validate_iso_date(updates.start_time):
                raise ValueError("Invalid update_event updates.start_time")
            return GatewayToolCall(
                tool_name=tool_call.tool_name,
                arguments=args.model_dump(mode="json", exclude_none=True),
            )

        if tool_call.tool_name == GatewayToolName.DELETE_EVENT:
            args = DeleteEventParams(**raw_args)
            return GatewayToolCall(
                tool_name=tool_call.tool_name,
                arguments=args.model_dump(mode="json", exclude_none=True),
            )

        raise ValueError(f"Unsupported tool: {tool_call.tool_name}")

    def _sanitize_state_patch(self, raw_patch: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only state fields that the agent is allowed to mutate."""
        if not isinstance(raw_patch, dict):
            return {}

        sanitized: Dict[str, Any] = {}
        working_state = raw_patch.get("working_state")
        if isinstance(working_state, dict):
            allowed_working_state: Dict[str, Any] = {}

            for key in ("goal", "status", "plan_summary"):
                value = working_state.get(key)
                if value is None or isinstance(value, str):
                    if key in working_state:
                        allowed_working_state[key] = value

            for key in ("pending_action", "resolved_entities"):
                value = working_state.get(key)
                if isinstance(value, dict):
                    allowed_working_state[key] = value

            if allowed_working_state:
                sanitized["working_state"] = allowed_working_state

        return sanitized

    def _normalize_step_json(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(raw_data or {})

        if not isinstance(data.get("response_payload"), dict):
            data["response_payload"] = {}

        if not isinstance(data.get("state_patch"), dict):
            data["state_patch"] = {}
        else:
            data["state_patch"] = self._sanitize_state_patch(data["state_patch"])

        if "next_gateway_action" not in data:
            root_tool_call = data.get("tool_call")
            if isinstance(root_tool_call, dict):
                data["next_gateway_action"] = {"type": "tool_call", "tool_call": root_tool_call}
            else:
                data["next_gateway_action"] = {"type": "none", "tool_call": None}

        gateway_action = data.get("next_gateway_action")
        if isinstance(gateway_action, dict):
            if "type" not in gateway_action:
                gateway_action["type"] = "tool_call" if gateway_action.get("tool_call") else "none"
            if "tool_call" not in gateway_action:
                gateway_action["tool_call"] = None
            data["next_gateway_action"] = gateway_action
        else:
            data["next_gateway_action"] = {"type": "none", "tool_call": None}

        if "response_type" not in data:
            if data["next_gateway_action"].get("type") == GatewayActionType.TOOL_CALL.value:
                data["response_type"] = AgentResponseType.TOOL_CALL.value
            elif data.get("assistant_message"):
                data["response_type"] = AgentResponseType.FINISH.value
            else:
                data["response_type"] = AgentResponseType.CLARIFY.value

        if "assistant_message" not in data:
            data["assistant_message"] = None

        return data

    def _fallback_step_response(self, message: str, status: str = "awaiting_user") -> StepResponse:
        return StepResponse(
            response_type=AgentResponseType.CLARIFY,
            assistant_message=message,
            response_payload={},
            next_gateway_action=GatewayAction(type=GatewayActionType.NONE, tool_call=None),
            state_patch={"working_state": {"status": status}},
        )

    def _is_delete_intent(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            token in lowered
            for token in (
                "удали",
                "удалить",
                "удаление",
                "отмени",
                "отменить",
                "убери",
                "убрать",
                "снеси",
                "delete",
                "remove",
            )
        )

    def _is_split_task_request(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            token in lowered
            for token in (
                "разбей",
                "разбить",
                "декомпоз",
                "подзадач",
                "шаги",
                "план по шагам",
            )
        )

    def _default_legacy_reply(self, tool_name: ToolName, fallback: Optional[str] = None) -> str:
        if fallback:
            return fallback

        defaults = {
            ToolName.CREATE_EVENT: "Принято. Создаю событие.",
            ToolName.UPDATE_EVENT: "Принято. Обновляю событие.",
            ToolName.DELETE_EVENT: "Принято. Удаляю событие.",
            ToolName.FIND_FREE_SLOT: "Есть несколько подходящих вариантов.",
            ToolName.SUMMARIZE_WEEK: "Готово.",
            ToolName.SPLIT_TASK: "План готов.",
            ToolName.GENERAL_CHAT: "Готово.",
            ToolName.CLARIFICATION_NEEDED: "Уточни, пожалуйста, детали.",
        }
        return defaults.get(tool_name, "Принято. Обрабатываю запрос.")

    def _extract_legacy_goal(
            self,
            text: str,
            conversation: Optional[ConversationContext],
            state_patch: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        if isinstance(state_patch, dict):
            working_state = state_patch.get("working_state")
            if isinstance(working_state, dict):
                goal = working_state.get("goal")
                if isinstance(goal, str):
                    return goal

        if conversation and conversation.pending_action:
            pending_tool = self._tool_name_to_string(conversation.pending_action.tool_name)
            if pending_tool:
                return pending_tool

        if self._is_delete_intent(text):
            return ToolName.DELETE_EVENT.value

        if self._is_split_task_request(text):
            return ToolName.SPLIT_TASK.value

        return None

    def _gateway_write_to_legacy_tool(self, gateway_tool_name: GatewayToolName) -> Optional[ToolName]:
        mapping = {
            GatewayToolName.CREATE_EVENT: ToolName.CREATE_EVENT,
            GatewayToolName.UPDATE_EVENT: ToolName.UPDATE_EVENT,
            GatewayToolName.DELETE_EVENT: ToolName.DELETE_EVENT,
        }
        return mapping.get(gateway_tool_name)

    def _events_to_observation_items(self, events: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for event in events or []:
            if hasattr(event, "model_dump"):
                data = event.model_dump(mode="json", exclude_none=True)
            elif isinstance(event, dict):
                data = dict(event)
            else:
                continue

            item = {
                "id": data.get("id"),
                "summary": data.get("summary") or data.get("title"),
                "start": data.get("start"),
                "end": data.get("end"),
                "status": data.get("status"),
                "location": data.get("location"),
                "description": data.get("description"),
            }
            items.append({key: value for key, value in item.items() if value is not None})
        return items

    def _search_config_from_gateway(
            self,
            tool_name: GatewayToolName,
            arguments: Dict[str, Any],
    ) -> Optional[SearchConfig]:
        if tool_name == GatewayToolName.GET_FREE_SLOTS:
            start = arguments.get("start")
            end = arguments.get("end")
            if isinstance(start, str) and isinstance(end, str):
                return SearchConfig(
                    start=start,
                    end=end,
                    min_duration_minutes=int(arguments.get("min_duration_minutes", 30)),
                )
            return None

        if tool_name == GatewayToolName.LIST_EVENTS:
            start = arguments.get("start")
            end = arguments.get("end")
            if isinstance(start, str) and isinstance(end, str):
                return SearchConfig(start=start, end=end)
            return None

        if tool_name == GatewayToolName.FIND_EVENT:
            start = arguments.get("time_min")
            end = arguments.get("time_max")
            if isinstance(start, str) and isinstance(end, str):
                return SearchConfig(start=start, end=end)
            return None

        return None

    def _build_legacy_execute_state(self, request: ExecuteRequest) -> AgentState:
        state = self._conversation_to_state(request.conversation)
        legacy_goal = self._tool_name_to_string(request.tool_name)

        if not state.working_state.goal and legacy_goal:
            state.working_state.goal = legacy_goal

        if not state.working_state.pending_action and legacy_goal:
            state.working_state.pending_action = {
                "tool_name": legacy_goal,
                "stage": "execute",
                "status": "in_progress",
            }

        observation_tool: Optional[GatewayToolName] = None
        observation_result: Optional[Dict[str, Any]] = None

        if request.tool_name == ToolName.FIND_FREE_SLOT:
            slots = [
                slot.model_dump(mode="json", exclude_none=True)
                if hasattr(slot, "model_dump")
                else slot
                for slot in (request.fetched_slots or [])
                if isinstance(slot, dict) or hasattr(slot, "model_dump")
            ]
            observation_tool = GatewayToolName.GET_FREE_SLOTS
            observation_result = {"slots": slots}

        elif request.tool_name == ToolName.SUMMARIZE_WEEK:
            items = self._events_to_observation_items(request.fetched_events or [])
            observation_tool = GatewayToolName.LIST_EVENTS
            observation_result = {"items": items, "total_found": len(items)}

        elif request.tool_name in {ToolName.UPDATE_EVENT, ToolName.DELETE_EVENT}:
            items = self._events_to_observation_items(request.fetched_events or [])
            observation_tool = GatewayToolName.FIND_EVENT
            observation_result = {"items": items, "total_found": len(items)}

        if observation_tool and observation_result is not None:
            state.tool_observations.append(
                ToolObservation(
                    tool_name=observation_tool,
                    status=ActionStatus.SUCCESS,
                    result=observation_result,
                )
            )

        return state

    def _convert_step_to_legacy_analyze(
            self,
            response: StepResponse,
            request: AnalyzeRequest,
    ) -> Optional[AnalyzeResponse]:
        goal = self._extract_legacy_goal(request.text, request.conversation, response.state_patch)

        if response.response_type == AgentResponseType.TOOL_CALL:
            tool_call = response.next_gateway_action.tool_call
            if response.next_gateway_action.type != GatewayActionType.TOOL_CALL or not tool_call:
                return None

            if tool_call.tool_name in {
                GatewayToolName.GET_FREE_SLOTS,
                GatewayToolName.LIST_EVENTS,
                GatewayToolName.FIND_EVENT,
            }:
                search = self._search_config_from_gateway(tool_call.tool_name, tool_call.arguments)
                if not search:
                    return None

                if tool_call.tool_name == GatewayToolName.GET_FREE_SLOTS:
                    return AnalyzeResponse(
                        tool_name=ToolName.FIND_FREE_SLOT,
                        requirement=DataRequirementType.SLOTS,
                        data_params=search,
                    )

                if tool_call.tool_name == GatewayToolName.FIND_EVENT and goal not in {
                    ToolName.UPDATE_EVENT.value,
                    ToolName.DELETE_EVENT.value,
                }:
                    return None

                if goal == ToolName.DELETE_EVENT.value:
                    legacy_tool = ToolName.DELETE_EVENT
                elif tool_call.tool_name == GatewayToolName.LIST_EVENTS and goal not in {
                    ToolName.UPDATE_EVENT.value,
                    ToolName.DELETE_EVENT.value,
                }:
                    legacy_tool = ToolName.SUMMARIZE_WEEK
                else:
                    legacy_tool = ToolName.UPDATE_EVENT

                return AnalyzeResponse(
                    tool_name=legacy_tool,
                    requirement=DataRequirementType.EVENTS,
                    data_params=search,
                )

            legacy_tool = self._gateway_write_to_legacy_tool(tool_call.tool_name)
            if not legacy_tool:
                return None

            final_response = self.llm._validate_ml_response(
                MLResponse(
                    tool_name=legacy_tool,
                    reply_text=self._default_legacy_reply(legacy_tool),
                    parameters=tool_call.arguments,
                )
            )
            return AnalyzeResponse(
                tool_name=legacy_tool,
                requirement=DataRequirementType.NONE,
                final_response=final_response,
            )

        if response.response_type == AgentResponseType.CLARIFY:
            return AnalyzeResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                requirement=DataRequirementType.NONE,
                final_response=MLResponse(
                    tool_name=ToolName.CLARIFICATION_NEEDED,
                    reply_text=self._default_legacy_reply(
                        ToolName.CLARIFICATION_NEEDED,
                        response.assistant_message,
                    ),
                    parameters={},
                ),
            )

        if response.response_type == AgentResponseType.FINISH:
            payload = response.response_payload if isinstance(response.response_payload, dict) else {}
            if self._is_split_task_request(request.text):
                if not payload:
                    return None
                try:
                    params = SplitTaskParams(**payload)
                except Exception:
                    return None
                return AnalyzeResponse(
                    tool_name=ToolName.SPLIT_TASK,
                    requirement=DataRequirementType.NONE,
                    final_response=self.llm._validate_ml_response(
                        MLResponse(
                            tool_name=ToolName.SPLIT_TASK,
                            reply_text=self._default_legacy_reply(ToolName.SPLIT_TASK, response.assistant_message),
                            parameters=params,
                        )
                    ),
                )

            return AnalyzeResponse(
                tool_name=ToolName.GENERAL_CHAT,
                requirement=DataRequirementType.NONE,
                final_response=MLResponse(
                    tool_name=ToolName.GENERAL_CHAT,
                    reply_text=self._default_legacy_reply(ToolName.GENERAL_CHAT, response.assistant_message),
                    parameters={},
                ),
            )

        return None

    def _convert_step_to_legacy_execute(
            self,
            response: StepResponse,
            request: ExecuteRequest,
    ) -> Optional[MLResponse]:
        if response.response_type == AgentResponseType.TOOL_CALL:
            tool_call = response.next_gateway_action.tool_call
            if response.next_gateway_action.type != GatewayActionType.TOOL_CALL or not tool_call:
                return None

            legacy_tool = self._gateway_write_to_legacy_tool(tool_call.tool_name)
            if not legacy_tool:
                return None

            return self.llm._validate_ml_response(
                MLResponse(
                    tool_name=legacy_tool,
                    reply_text=self._default_legacy_reply(legacy_tool),
                    parameters=tool_call.arguments,
                )
            )

        if response.response_type == AgentResponseType.CLARIFY:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text=self._default_legacy_reply(
                    ToolName.CLARIFICATION_NEEDED,
                    response.assistant_message,
                ),
                parameters={},
            )

        if response.response_type == AgentResponseType.FINISH:
            if request.tool_name == ToolName.FIND_FREE_SLOT:
                ranked = self.llm.ranker.rank_slots(request.fetched_slots or [], request.context)
                return self.llm._validate_ml_response(
                    MLResponse(
                        tool_name=ToolName.FIND_FREE_SLOT,
                        reply_text=self._default_legacy_reply(
                            ToolName.FIND_FREE_SLOT,
                            response.assistant_message,
                        ),
                        parameters=RankedSlotParams(
                            ranked_slots=ranked[:3],
                            reasoning="Selected by AI Ranking (Work hours priority)",
                        ),
                    )
                )

            if request.tool_name == ToolName.SUMMARIZE_WEEK:
                return MLResponse(
                    tool_name=ToolName.SUMMARIZE_WEEK,
                    reply_text=self._default_legacy_reply(
                        ToolName.SUMMARIZE_WEEK,
                        response.assistant_message,
                    ),
                    parameters={},
                )

            if request.tool_name == ToolName.SPLIT_TASK:
                payload = response.response_payload if isinstance(response.response_payload, dict) else {}
                if not payload:
                    return None
                try:
                    params = SplitTaskParams(**payload)
                except Exception:
                    return None
                return self.llm._validate_ml_response(
                    MLResponse(
                        tool_name=ToolName.SPLIT_TASK,
                        reply_text=self._default_legacy_reply(
                            ToolName.SPLIT_TASK,
                            response.assistant_message,
                        ),
                        parameters=params,
                    )
                )

            if request.tool_name in {ToolName.GENERAL_CHAT, ToolName.CLARIFICATION_NEEDED}:
                return MLResponse(
                    tool_name=request.tool_name,
                    reply_text=self._default_legacy_reply(request.tool_name, response.assistant_message),
                    parameters={},
                )

        return None

    def legacy_analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        try:
            step_response = self.step(
                StepRequest(
                    text=request.text,
                    context=request.context,
                    conversation=request.conversation,
                    all_context=request.all_context,
                )
            )
            converted = self._convert_step_to_legacy_analyze(step_response, request)
            if converted is not None:
                return converted
            logger.warning("legacy analyze adapter returned no compatible step response; using legacy fallback")
        except Exception as exc:
            logger.warning(f"legacy analyze adapter failed: {exc}")

        return self.llm.analyze(request)

    def legacy_execute(self, request: ExecuteRequest) -> MLResponse:
        try:
            state = self._build_legacy_execute_state(request)
            step_response = self.step(
                StepRequest(
                    text=request.text,
                    context=request.context,
                    state=state,
                    all_context=request.all_context,
                )
            )
            converted = self._convert_step_to_legacy_execute(step_response, request)
            if converted is not None:
                return converted
            logger.warning("legacy execute adapter returned no compatible step response; using legacy fallback")
        except Exception as exc:
            logger.warning(f"legacy execute adapter failed: {exc}")

        return self.llm.execute(request)

    def step(self, request: StepRequest) -> StepResponse:
        state = self._normalize_state(request)
        state_for_prompt = self._prepare_state_for_prompt(state, request.context)

        system_prompt = p.STEP_SYSTEM_PROMPT.format(
            persona=p.CHAT_PERSONA,
        )
        user_payload = self._build_step_payload(request, state_for_prompt)
        runtime_context = self.llm.build_runtime_context(request.context)

        logger.info(f"step: {request.text[:60] if request.text else '[no new text]'}")
        raw_data = self.llm._call_model(system_prompt, user_payload, runtime_context=runtime_context)

        if not raw_data:
            return self._fallback_step_response(
                "Не получилось надёжно обработать шаг. Уточни, пожалуйста, что именно нужно сделать."
            )
        if not isinstance(raw_data, dict):
            return self._fallback_step_response(
                "Я получил неожиданный формат ответа. Уточни, пожалуйста, задачу ещё раз."
            )

        normalized = self._normalize_step_json(raw_data)

        try:
            response = StepResponse(**normalized)

            if response.response_type == AgentResponseType.TOOL_CALL:
                if response.next_gateway_action.type != GatewayActionType.TOOL_CALL:
                    raise ValueError("tool_call response must set next_gateway_action.type=tool_call")
                if not response.next_gateway_action.tool_call:
                    raise ValueError("tool_call response must contain next_gateway_action.tool_call")
                response.next_gateway_action = GatewayAction(
                    type=GatewayActionType.TOOL_CALL,
                    tool_call=self._validate_tool_call(response.next_gateway_action.tool_call),
                )
                return response

            response.next_gateway_action = GatewayAction(type=GatewayActionType.NONE, tool_call=None)
            if not response.assistant_message:
                if response.response_type == AgentResponseType.CLARIFY:
                    response.assistant_message = "Уточни, пожалуйста, что именно ты хочешь сделать."
                else:
                    response.assistant_message = "Готово."
            return response

        except Exception as exc:
            logger.warning(f"step validation err: {exc}")
            return self._fallback_step_response(
                "Мне нужно чуть больше ясности, чтобы продолжить. Уточни, пожалуйста, задачу ещё раз."
            )
