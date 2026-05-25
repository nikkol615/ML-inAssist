"""
 Модуль: llm.py
 Назначение: Сервис взаимодействия с языковой моделью (LLM)
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

import json
import os
import logging
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from llama_cpp import LlamaDiskCache
except ImportError:
    LlamaDiskCache = None

from pydantic import ValidationError

from app.schemas import (
    AnalyzeRequest, AnalyzeResponse, ExecuteRequest, MLResponse,
    ToolName, DataRequirementType, RankedSlotParams, SplitTaskParams,
    CreateEventParams, UpdateEventParams, DeleteEventParams
)
from app.services.ranking import RankingService
import app.prompts as p

logger = logging.getLogger("ml_llm")
logger.setLevel(logging.INFO)


class LLMService:
    def __init__(self):
        self.use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        self.local_model = None
        self.client = None
        self.prompt_cache_enabled = False
        self.prompt_cache_dir: Optional[Path] = None

        if self.use_local:
            model_path = os.getenv(
                "LOCAL_MODEL_PATH",
                "/app/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
            )

            if Llama is None:
                logger.error("llama-cpp-python not installed")
            elif not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                logger.error("   Make sure the model was downloaded by model-downloader")
            else:
                try:
                    model_size_gb = os.path.getsize(model_path) / (1024 ** 3)
                    logger.info(f"Loading LLM model ({model_size_gb:.1f}GB): {model_path}")
                    logger.info("   This may take 1-3 minutes on first load...")
                    start_time = time.time()
                    self.local_model = Llama(
                        model_path=model_path,
                        n_ctx=8192,
                        n_threads=4,
                        n_gpu_layers=0,
                        verbose=False
                    )
                    load_time = time.time() - start_time
                    logger.info(f"LLM model loaded successfully in {load_time:.1f}s")
                    self._configure_prompt_cache(model_path)
                except Exception as e:
                    logger.error(f"LLM load failed: {e}")
        else:
            self.api_key = os.getenv("GROQ_API_KEY")

            if Groq and self.api_key:
                try:
                    self.client = Groq(api_key=self.api_key)
                    self.model_name = "llama-3.3-70b-versatile"
                    logger.info(f"groq ok ({self.model_name})")
                except Exception as e:
                    logger.error(f"groq failed: {e}")
            else:
                logger.warning("no api key")

        self.ranker = RankingService()

        self.max_retries = 1

        self.json_grammar = None
        if self.use_local and self.local_model:
            try:
                from llama_cpp import LlamaGrammar
                # BNF-грамматика: ограничивает выход модели только валидным JSON.
                # Каждый токен проверяется на соответствие правилам до генерации.
                self.json_grammar = LlamaGrammar.from_string(r'''
root ::= object
object ::= "{" ws members? ws "}"
members ::= pair (ws "," ws pair)*
pair ::= string ws ":" ws value
value ::= string | number | object | array | "true" | "false" | "null"
array ::= "[" ws elements? ws "]"
elements ::= value (ws "," ws value)*
string ::= "\"" characters "\""
characters ::= character*
character ::= [^"\\] | "\\" escape
escape ::= ["\\nrt/]
number ::= "-"? digits ("." digits)? ([eE] [+-]? digits)?
digits ::= [0-9]+
ws ::= [ \t\n\r]*
''')
                logger.info("JSON grammar loaded")
            except Exception as e:
                logger.warning(f"JSON grammar failed: {e}, using fallback")
            self._warm_prompt_cache()

    def _prompt_cache_requested(self) -> bool:
        value = os.getenv("LLAMA_PROMPT_CACHE", "true").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _prompt_cache_capacity_bytes(self) -> int:
        raw_value = os.getenv("LLAMA_CACHE_SIZE_BYTES", str(1024 ** 3))
        try:
            return max(int(raw_value), 64 * 1024 * 1024)
        except ValueError:
            return 1024 ** 3

    def _prompt_cache_root(self) -> Path:
        configured = os.getenv("LLAMA_CACHE_DIR")
        if configured:
            return Path(configured)

        data_root = Path("/app/data") if Path("/app/data").exists() else Path("data")
        return data_root / "llama_cache"

    def _model_cache_key(self, model_path: str) -> str:
        resolved_path = os.path.abspath(model_path)
        try:
            stat = os.stat(resolved_path)
            signature = (
                f"{resolved_path}|{stat.st_size}|{int(stat.st_mtime)}|"
                f"{p.PROMPT_CACHE_VERSION}"
            )
        except OSError:
            signature = f"{resolved_path}|{p.PROMPT_CACHE_VERSION}"
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]

    def _configure_prompt_cache(self, model_path: str) -> None:
        """Attach llama.cpp disk cache scoped to the current model file."""
        if not self._prompt_cache_requested():
            logger.info("LLM prompt cache disabled")
            return
        if LlamaDiskCache is None:
            logger.warning("LlamaDiskCache is not available; prompt cache disabled")
            return

        try:
            cache_dir = self._prompt_cache_root() / self._model_cache_key(model_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = LlamaDiskCache(
                cache_dir=str(cache_dir),
                capacity_bytes=self._prompt_cache_capacity_bytes(),
            )
            self.local_model.set_cache(cache)
            self.prompt_cache_dir = cache_dir
            self.prompt_cache_enabled = True
            logger.info(f"LLM prompt cache enabled: {cache_dir}")
        except Exception as exc:
            self.prompt_cache_enabled = False
            self.prompt_cache_dir = None
            logger.warning(f"LLM prompt cache setup failed: {exc}")

    @staticmethod
    def _format_llama_prompt(system_text: str, user_text: str) -> str:
        # llama-cpp добавляет BOS-токен автоматически — не дублируем вручную
        return f"""<|start_header_id|>system<|end_header_id|>

{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def build_runtime_context(self, context: Any, dynamic_input: Optional[Dict[str, Any]] = None) -> str:
        if hasattr(context, "model_dump"):
            context_data = context.model_dump(mode="json")
        elif isinstance(context, dict):
            context_data = context
        else:
            context_data = {}

        parts = [
            p.RUNTIME_CONTEXT_PROMPT.format(
                current_time=context_data.get("current_time") or "",
                timezone=context_data.get("timezone") or "UTC",
            ).strip()
        ]

        if dynamic_input is not None:
            parts.append(
                p.DYNAMIC_INPUT_PROMPT.format(
                    dynamic_input_json=json.dumps(
                        {"dynamic_input": dynamic_input},
                        ensure_ascii=False,
                    )
                ).strip()
            )

        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _build_model_user_text(user_text: str, runtime_context: Optional[str]) -> str:
        runtime_context = (runtime_context or "").strip()
        if not runtime_context:
            return user_text
        return f"{runtime_context}\n\nUSER PAYLOAD:\n{user_text}"

    def _warm_prompt_cache(self) -> None:
        """Preload stable router and step prompt prefixes into the llama.cpp cache."""
        if not (self.use_local and self.local_model and self.prompt_cache_enabled):
            return

        try:
            runtime_context = self.build_runtime_context(
                {"current_time": "", "timezone": "UTC"},
                dynamic_input={},
            )
            warm_user_text = self._build_model_user_text("{}", runtime_context)
            for system_prompt in (
                p.ROUTER_SYSTEM_PROMPT.format(persona=p.CHAT_PERSONA),
                p.STEP_SYSTEM_PROMPT.format(persona=p.CHAT_PERSONA),
            ):
                self.local_model(
                    self._format_llama_prompt(system_prompt, warm_user_text),
                    max_tokens=1,
                    temperature=0.0,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                    echo=False,
                )
            logger.info("LLM prompt cache warmed")
        except Exception as exc:
            logger.warning(f"LLM prompt cache warmup failed: {exc}")

    def _call_model(
            self,
            system_text: str,
            user_text: str,
            retry_count: int = 0,
            runtime_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        result = None
        model_user_text = self._build_model_user_text(user_text, runtime_context)

        if self.use_local and self.local_model:
            result = self._call_local_model(system_text, model_user_text)
        elif self.client:
            result = self._call_groq_api(system_text, model_user_text)
        else:
            logger.error("no model")
            return None

        if result is None and retry_count < self.max_retries:
            logger.warning(f"retry {retry_count + 1}/{self.max_retries}: invalid JSON response")
            retry_system = system_text + "\n\nIMPORTANT: Your previous response was not valid JSON. Please return ONLY valid JSON object."
            return self._call_model(
                retry_system,
                user_text,
                retry_count + 1,
                runtime_context=runtime_context,
            )

        return result

    def _call_local_model(self, system_text: str, user_text: str) -> Optional[Dict[str, Any]]:
        """Llama 3 Instruct через llama-cpp-python."""
        try:
            start_time = time.time()

            prompt = self._format_llama_prompt(system_text, user_text)

            output = self.local_model(
                prompt,
                max_tokens=1024,
                temperature=0.1,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False,
                grammar=self.json_grammar
            )

            elapsed = time.time() - start_time
            usage = output.get("usage", {}) if isinstance(output, dict) else {}
            total_tokens = usage.get("completion_tokens", 0) if usage else 0
            tokens_per_sec = total_tokens / elapsed if elapsed > 0.001 else 0
            logger.info(
                f"[TELEMETRY] local_llm: latency={elapsed:.2f}s, tokens={total_tokens}, tokens/sec={tokens_per_sec:.1f}")

            content = output["choices"][0]["text"].strip()
            if not content:
                raise ValueError("Empty response from local LLM")

            # Fallback если grammar не загружена: модель может вернуть
            # "Here is the JSON: {...}" — вырезаем только JSON часть
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]

            logger.debug(
                "[HISTORY] local_llm exchange: request_chars=%s response_chars=%s",
                len(user_text),
                len(content),
            )

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"bad json: {e}")
            return None
        except Exception as e:
            logger.error(f"local llm err: {e}")
            return None

    def _call_groq_api(self, system_text: str, user_text: str) -> Optional[Dict[str, Any]]:
        """Groq Cloud API c принудительным JSON mode."""
        try:
            start_time = time.time()

            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text}
                ],
                model=self.model_name,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1024
            )

            elapsed = time.time() - start_time
            # Groq SDK может вернуть None в usage при streaming или ошибке
            try:
                total_tokens = completion.usage.completion_tokens if completion.usage else 0
            except (AttributeError, TypeError):
                total_tokens = 0
            tokens_per_sec = total_tokens / elapsed if elapsed > 0.001 else 0
            logger.info(
                f"[TELEMETRY] groq_api: latency={elapsed:.2f}s, tokens={total_tokens}, tokens/sec={tokens_per_sec:.1f}")

            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            logger.debug(
                "[HISTORY] groq_api exchange: request_chars=%s response_chars=%s",
                len(user_text),
                len(content),
            )

            return json.loads(content)

        except json.JSONDecodeError:
            logger.error("bad json from api")
            return None
        except Exception as e:
            logger.error(f"api err: {e}")
            return None

    def _validate_iso_date(self, date_str: str) -> bool:
        """ISO-8601 валидация. Python <3.11 не понимает 'Z' как UTC."""
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    def _build_user_payload(
            self,
            text: str,
            context: Any,
            conversation: Any = None,
            all_context: Optional[str] = None
    ) -> str:
        """Единый формат user payload для runtime/training/eval."""
        if hasattr(context, "model_dump"):
            context_data = context.model_dump(mode="json")
        elif isinstance(context, dict):
            context_data = context
        else:
            context_data = {}

        if hasattr(conversation, "model_dump"):
            conversation_data = conversation.model_dump(mode="json", exclude_none=True)
        elif isinstance(conversation, dict):
            conversation_data = {k: v for k, v in conversation.items() if v is not None}
        else:
            conversation_data = None

        payload = {"text": text, "context": context_data}
        if conversation_data:
            payload["conversation"] = conversation_data
        if all_context:
            payload["all_context"] = all_context

        return json.dumps(payload, ensure_ascii=False)

    def _validate_execute_parameters(self, tool_name: ToolName, parameters: Any):
        """Приводим параметры к строгим схемам там, где они уже определены."""
        raw_params = (
            parameters.model_dump(mode="json")
            if hasattr(parameters, "model_dump")
            else (parameters or {})
        )

        if tool_name == ToolName.CREATE_EVENT:
            params = CreateEventParams(**raw_params)
            if not self._validate_iso_date(params.start_time):
                raise ValueError("Invalid create_event start_time")
            return params

        if tool_name == ToolName.UPDATE_EVENT:
            params = UpdateEventParams(**raw_params)
            updates = params.updates
            if not any([
                updates.title,
                updates.start_time,
                updates.duration_minutes is not None,
                updates.location,
                updates.description,
            ]):
                raise ValueError("update_event must contain at least one update field")
            if updates.start_time and not self._validate_iso_date(updates.start_time):
                raise ValueError("Invalid update_event updates.start_time")
            return params

        if tool_name == ToolName.DELETE_EVENT:
            return DeleteEventParams(**raw_params)

        if tool_name == ToolName.FIND_FREE_SLOT:
            return RankedSlotParams(**raw_params)

        if tool_name == ToolName.SPLIT_TASK:
            return SplitTaskParams(**raw_params)

        return raw_params if isinstance(raw_params, dict) else {}

    def _validate_ml_response(self, response: MLResponse) -> MLResponse:
        validated_params = self._validate_execute_parameters(
            response.tool_name,
            response.parameters
        )
        return MLResponse(
            tool_name=response.tool_name,
            reply_text=response.reply_text,
            parameters=validated_params
        )

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Router: классифицирует намерение и определяет, нужны ли данные из Google Calendar."""
        system_prompt = p.ROUTER_SYSTEM_PROMPT.format(
            persona=p.CHAT_PERSONA
        )
        user_payload = self._build_user_payload(
            request.text,
            request.context,
            request.conversation,
            request.all_context
        )
        runtime_context = self.build_runtime_context(request.context)

        logger.info(f"analyze: {request.text[:40]}")

        raw_data = self._call_model(system_prompt, user_payload, runtime_context=runtime_context)

        if not raw_data:
            return self._create_error_response("Ошибка связи с мозгом (API Error).")

        # Авто-коррекция JSON: LLM может вернуть tool_name на верхнем уровне
        # вместо final_response, или раскидать parameters по корню объекта.
        # Здесь собираем всё в правильную структуру.
        if "final_response" in raw_data and isinstance(raw_data["final_response"], dict):
            fr = raw_data["final_response"]

            if "tool_name" not in fr:
                fr["tool_name"] = raw_data.get("tool_name", "general_chat")

            if "reply_text" not in fr:
                fr["reply_text"] = "Принято. Обрабатываю запрос."

            if "parameters" not in fr:
                # LLM может положить title, start_time прямо в final_response
                # вместо {"parameters": {"title": ..., "start_time": ...}}
                # Собираем все неизвестные ключи в parameters
                reserved_keys = {"tool_name", "reply_text", "parameters"}
                extracted_params = {k: v for k, v in fr.items() if k not in reserved_keys}
                fr["parameters"] = extracted_params if extracted_params else {}

        elif raw_data.get("requirement") == DataRequirementType.NONE.value:
            root_reply = raw_data.get("reply_text")
            root_params = raw_data.get("parameters")
            if root_params is None:
                reserved_keys = {"tool_name", "requirement", "data_params", "final_response", "reply_text",
                                 "parameters"}
                extracted_params = {k: v for k, v in raw_data.items() if k not in reserved_keys}
                root_params = extracted_params if extracted_params else {}
            if root_reply is not None or root_params is not None:
                raw_data["final_response"] = {
                    "tool_name": raw_data.get("tool_name", "general_chat"),
                    "reply_text": root_reply or "Принято. Обрабатываю запрос.",
                    "parameters": root_params or {}
                }

        try:
            response = AnalyzeResponse(**raw_data)

            if response.requirement in {DataRequirementType.SLOTS, DataRequirementType.EVENTS}:
                if not response.data_params:
                    logger.warning("calendar data requested but no dates")
                    raise ValueError("Requirement needs data_params but they are missing")

                # Проверяем, что даты валидные (LLM иногда генерит мусор)
                if not (self._validate_iso_date(response.data_params.start) and
                        self._validate_iso_date(response.data_params.end)):
                    logger.warning(f"bad dates: {response.data_params}")
                    raise ValueError("Invalid ISO date format generated by LLM")

            if response.requirement == DataRequirementType.NONE:
                if not response.final_response:
                    response.final_response = MLResponse(
                        tool_name=response.tool_name,
                        reply_text="Принято. Обрабатываю запрос.",
                        parameters={}
                    )
                response.final_response = self._validate_ml_response(response.final_response)

            return response

        except (ValidationError, ValueError) as e:
            logger.warning(f"validation err: {e}")

            clarify_prompt = f"""
{p.CHAT_PERSONA}
TASK: The user's request could not be processed due to missing or invalid parameters.
Original request: "{request.text}"
Error: {str(e)[:100]}

Generate a friendly clarification request in Russian. Ask the user to rephrase or provide more details.
OUTPUT JSON: {{ "reply_text": "..." }}
"""
            clarify_resp = self._call_model(clarify_prompt, user_payload, runtime_context=runtime_context)
            reply_text = (clarify_resp.get("reply_text") if clarify_resp
                          else "Не совсем понял. Можете уточнить ваш запрос?")

            return AnalyzeResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                requirement=DataRequirementType.NONE,
                final_response=MLResponse(
                    tool_name=ToolName.CLARIFICATION_NEEDED,
                    reply_text=reply_text,
                    parameters={}
                )
            )

    def execute(self, request: ExecuteRequest) -> MLResponse:
        """Executor: обрабатывает данные Google Calendar и генерирует ответ."""
        logger.info(f"exec: {request.tool_name}")

        try:
            if request.tool_name == ToolName.FIND_FREE_SLOT:
                return self._handle_find_slot(request)

            if request.tool_name == ToolName.SUMMARIZE_WEEK:
                return self._handle_summarize(request)

            if request.tool_name == ToolName.SPLIT_TASK:
                return self._handle_split_task(request)

            if request.tool_name == ToolName.UPDATE_EVENT:
                return self._handle_update_event(request)

            if request.tool_name == ToolName.DELETE_EVENT:
                return self._handle_delete_event(request)

            return MLResponse(
                tool_name=ToolName.GENERAL_CHAT,
                reply_text="Данные обработаны, но я не знаю специфического обработчика.",
                parameters={}
            )

        except Exception as e:
            logger.error(f"exec failed: {e}")
            return MLResponse(
                tool_name=ToolName.GENERAL_CHAT,
                reply_text="Произошла техническая ошибка при обработке данных.",
                parameters={}
            )

    def _handle_find_slot(self, request: ExecuteRequest) -> MLResponse:
        """ML-ранжирование свободных слотов и генерация человечного ответа."""
        slots = request.fetched_slots or []
        ranked = self.ranker.rank_slots(slots, request.context)
        user_payload = self._build_user_payload(
            request.text,
            request.context,
            request.conversation,
            request.all_context
        )

        if not ranked:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="К сожалению, Google не нашел свободных слотов в указанном диапазоне.",
                parameters={}
            )

        best_time = ranked[0].start.split("T")[-1][:5]
        alt_time = ranked[1].start.split("T")[-1][:5] if len(ranked) > 1 else "другое"

        prompt = p.SLOT_FOUND_PROMPT.format(
            persona=p.CHAT_PERSONA
        )
        runtime_context = self.build_runtime_context(
            request.context,
            dynamic_input={
                "best_slot_time": best_time,
                "alt_slot_time": alt_time,
            },
        )

        llm_resp = self._call_model(prompt, user_payload, runtime_context=runtime_context)
        reply = llm_resp.get("reply_text",
                             f"Предлагаю время: {best_time}.") if llm_resp else f"Лучшее время: {best_time}"

        return self._validate_ml_response(MLResponse(
            tool_name=ToolName.FIND_FREE_SLOT,
            reply_text=reply,
            parameters=RankedSlotParams(
                ranked_slots=ranked[:3],
                reasoning="Selected by AI Ranking (Work hours priority)"
            )
        ))

    def _handle_summarize(self, request: ExecuteRequest) -> MLResponse:
        """LLM-саммари событий календаря."""
        events = request.fetched_events or []
        user_payload = self._build_user_payload(
            request.text,
            request.context,
            request.conversation,
            request.all_context
        )
        events_str = json.dumps([
            {"title": e.title, "start": e.start} for e in events
        ], ensure_ascii=False)

        prompt = p.SUMMARIZE_PROMPT.format(
            persona=p.CHAT_PERSONA
        )
        runtime_context = self.build_runtime_context(
            request.context,
            dynamic_input={
                "user_request": request.text,
                "events": json.loads(events_str),
            },
        )

        raw_data = self._call_model(prompt, user_payload, runtime_context=runtime_context)
        reply = raw_data.get("reply_text", "События проанализированы.") if raw_data else "Готово."

        return MLResponse(
            tool_name=ToolName.SUMMARIZE_WEEK,
            reply_text=reply,
            parameters={}
        )

    def _handle_update_event(self, request: ExecuteRequest) -> MLResponse:
        """LLM-определение какое событие изменить и как."""
        events = request.fetched_events or []
        user_payload = self._build_user_payload(
            request.text,
            request.context,
            request.conversation,
            request.all_context
        )

        if not events:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="Не нашёл события для изменения. Уточните, какое событие вы хотите изменить?",
                parameters={}
            )

        events_str = json.dumps([
            {
                "id": e.id,
                "title": e.title,
                "start": e.start,
                "end": e.end,
                "location": e.location,
                "description": e.description,
            }
            for e in events
        ], ensure_ascii=False)

        prompt = p.UPDATE_EVENT_PROMPT.format(
            persona=p.CHAT_PERSONA
        )
        runtime_context = self.build_runtime_context(
            request.context,
            dynamic_input={
                "user_request": request.text,
                "events": json.loads(events_str),
            },
        )

        raw_data = self._call_model(prompt, user_payload, runtime_context=runtime_context)

        if raw_data:
            reply = raw_data.get("reply_text", "Готово, событие обновлено.")
            params = raw_data.get("parameters", {})
        else:
            reply = "Событие найдено. Что именно нужно изменить?"
            params = {}

        try:
            return self._validate_ml_response(MLResponse(
                tool_name=ToolName.UPDATE_EVENT,
                reply_text=reply,
                parameters=params
            ))
        except (ValidationError, ValueError):
            logger.warning("update_event bad schema")
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="Я нашёл событие, но не смог надёжно сформировать обновление. Уточните новое время, название или длительность.",
                parameters={}
            )

    def _handle_split_task(self, request: ExecuteRequest) -> MLResponse:
        """LLM-разбиение большой задачи на подзадачи с оценкой времени."""
        prompt = p.SPLIT_TASK_PROMPT.format(
            persona=p.CHAT_PERSONA
        )
        user_payload = self._build_user_payload(
            request.text,
            request.context,
            request.conversation,
            request.all_context
        )
        runtime_context = self.build_runtime_context(
            request.context,
            dynamic_input={"user_request": request.text},
        )

        raw_data = self._call_model(prompt, user_payload, runtime_context=runtime_context)

        try:
            if not raw_data:
                raise ValueError("Empty LLM response")

            reply = raw_data.get("reply_text", "План готов.")
            params_dict = raw_data.get("parameters", {})
            valid_params = SplitTaskParams(**params_dict)

            return self._validate_ml_response(MLResponse(
                tool_name=ToolName.SPLIT_TASK,
                reply_text=reply,
                parameters=valid_params
            ))
        except ValidationError:
            logger.warning("split_task bad schema")
            return MLResponse(
                tool_name=ToolName.GENERAL_CHAT,
                reply_text="Я составил план, но возникла ошибка форматирования. Давайте попробуем разбить задачу вручную?",
                parameters={}
            )

    def _handle_delete_event(self, request: ExecuteRequest) -> MLResponse:
        """Fallback-обработчик legacy delete_event, когда событие уже найдено gateway."""
        events = request.fetched_events or []

        if not events:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="Не нашёл событие для удаления. Уточните, какое событие нужно удалить?",
                parameters={}
            )

        if len(events) > 1:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="Нашёл несколько событий. Уточните, какое именно нужно удалить.",
                parameters={}
            )

        target = events[0]
        if not target.id:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="Нашёл событие, но не смог определить его идентификатор для удаления. Уточните запрос.",
                parameters={}
            )

        title = target.title.strip() if isinstance(target.title, str) else ""
        reply = f'Принято. Удаляю событие "{title}".' if title else "Принято. Удаляю событие."

        return self._validate_ml_response(MLResponse(
            tool_name=ToolName.DELETE_EVENT,
            reply_text=reply,
            parameters=DeleteEventParams(event_id=target.id)
        ))

    def _create_error_response(self, msg: str) -> AnalyzeResponse:
        """Формируем стандартный ответ об ошибке"""
        return AnalyzeResponse(
            tool_name=ToolName.GENERAL_CHAT,
            requirement=DataRequirementType.NONE,
            final_response=MLResponse(
                tool_name=ToolName.GENERAL_CHAT,
                reply_text=msg,
                parameters={}
            )
        )
