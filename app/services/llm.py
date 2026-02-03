"""
 Модуль: llm.py
 Назначение: Сервис взаимодействия с языковой моделью (LLM)
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════════════════════

import json
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from pydantic import ValidationError

from app.schemas import (
    AnalyzeRequest, AnalyzeResponse, ExecuteRequest, MLResponse,
    ToolName, DataRequirementType, RankedSlotParams, SplitTaskParams
)
from app.services.ranking import RankingService
import app.prompts as p


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("ml_llm")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# КЛАСС LLMService
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    def __init__(self):
        # Определяем режим: локальная модель или API
        self.use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        self.local_model = None
        self.client = None
        
        if self.use_local:
            model_path = os.getenv(
                "LOCAL_MODEL_PATH", 
                "/app/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
            )
            
            if Llama is None:
                logger.error("❌ llama-cpp-python not installed")
            elif not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                logger.error("   Make sure the model was downloaded by model-downloader")
            else:
                try:
                    model_size_gb = os.path.getsize(model_path) / (1024**3)
                    logger.info(f"🔄 Loading LLM model ({model_size_gb:.1f}GB): {model_path}")
                    logger.info("   This may take 1-3 minutes on first load...")
                    start_time = time.time()
                    self.local_model = Llama(
                        model_path=model_path,
                        n_ctx=4096,   # Размер контекстного окна (макс. токенов в диалоге)
                        n_threads=4,  # CPU потоки для параллельной обработки
                        n_gpu_layers=0,  # 0 = CPU, >0 = слои на GPU, -1 = всё на GPU
                        verbose=False
                    )
                    load_time = time.time() - start_time
                    logger.info(f"✓ LLM model loaded successfully in {load_time:.1f}s")
                except Exception as e:
                    logger.error(f"❌ LLM load failed: {e}")
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
        
        # JSON грамматика для llama-cpp
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

    # -------------------------------------------------------------------------
    # ВЗАИМОДЕЙСТВИЕ С МОДЕЛИ
    # -------------------------------------------------------------------------

    def _call_model(self, system_text: str, user_text: str, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        result = None
        
        if self.use_local and self.local_model:
            result = self._call_local_model(system_text, user_text)
        elif self.client:
            result = self._call_groq_api(system_text, user_text)
        else:
            logger.error("no model")
            return None
        
        # Retry при невалидном ответе
        if result is None and retry_count < self.max_retries:
            logger.warning(f"retry {retry_count + 1}/{self.max_retries}: invalid JSON response")
            # Добавляем указание на ошибку в промпт для retry
            retry_system = system_text + "\n\nIMPORTANT: Your previous response was not valid JSON. Please return ONLY valid JSON object."
            return self._call_model(retry_system, user_text, retry_count + 1)
        
        return result

    def _call_local_model(self, system_text: str, user_text: str) -> Optional[Dict[str, Any]]:
        """Llama 3 Instruct через llama-cpp-python."""
        try:
            start_time = time.time()
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            
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
            logger.info(f"[TELEMETRY] local_llm: latency={elapsed:.2f}s, tokens={total_tokens}, tokens/sec={tokens_per_sec:.1f}")
            
            content = output["choices"][0]["text"].strip()
            if not content:
                raise ValueError("Empty response from local LLM")
            
            # Fallback если grammar не загружена: модель может вернуть
            # "Here is the JSON: {...}" — вырезаем только JSON часть
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                content = content[json_start:json_end]
            
            logger.debug(f"[HISTORY] request: {user_text[:100]}... | response: {content[:100]}...")
            
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
            logger.info(f"[TELEMETRY] groq_api: latency={elapsed:.2f}s, tokens={total_tokens}, tokens/sec={tokens_per_sec:.1f}")

            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")

            logger.debug(f"[HISTORY] request: {user_text[:100]}... | response: {content[:100]}...")

            return json.loads(content)

        except json.JSONDecodeError:
            logger.error("bad json from api")
            return None
        except Exception as e:
            logger.error(f"api err: {e}")
            return None

    # -------------------------------------------------------------------------
    # ВАЛИДАЦИЯ И ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # -------------------------------------------------------------------------

    def _validate_iso_date(self, date_str: str) -> bool:
        """ISO-8601 валидация. Python <3.11 не понимает 'Z' как UTC."""
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    # -------------------------------------------------------------------------
    # ОСНОВНЫЕ МЕТОДЫ API
    # -------------------------------------------------------------------------

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        """Router: классифицирует намерение и определяет, нужны ли данные из Google Calendar."""
        system_prompt = p.ROUTER_SYSTEM_PROMPT.format(
            persona=p.CHAT_PERSONA,
            current_time=request.context.current_time,
            timezone=request.context.timezone
        )

        logger.info(f"analyze: {request.text[:40]}")

        raw_data = self._call_model(system_prompt, request.text)

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

        try:
            response = AnalyzeResponse(**raw_data)

            if response.requirement == DataRequirementType.SLOTS:
                if not response.data_params:
                    logger.warning("slots requested but no dates")
                    raise ValueError("Requirement is SLOTS but data_params is missing")

                # Проверяем, что даты валидные (LLM иногда генерит мусор)
                if not (self._validate_iso_date(response.data_params.start) and
                        self._validate_iso_date(response.data_params.end)):
                    logger.warning(f"bad dates: {response.data_params}")
                    raise ValueError("Invalid ISO date format generated by LLM")

            return response

        except (ValidationError, ValueError) as e:
            logger.warning(f"validation err: {e}")

            # Просим LLM сформулировать уточняющий вопрос на основе исходного запроса
            clarify_prompt = f"""
{p.CHAT_PERSONA}
TASK: The user's request could not be processed due to missing or invalid parameters.
Original request: "{request.text}"
Error: {str(e)[:100]}

Generate a friendly clarification request in Russian. Ask the user to rephrase or provide more details.
OUTPUT JSON: {{ "reply_text": "..." }}
"""
            clarify_resp = self._call_model(clarify_prompt, request.text)
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
            # Диспетчеризация по типу инструмента
            if request.tool_name == ToolName.FIND_FREE_SLOT:
                return self._handle_find_slot(request)

            if request.tool_name == ToolName.SUMMARIZE_WEEK:
                return self._handle_summarize(request)

            if request.tool_name == ToolName.SPLIT_TASK:
                return self._handle_split_task(request)

            if request.tool_name == ToolName.UPDATE_EVENT:
                return self._handle_update_event(request)

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

    # -------------------------------------------------------------------------
    # ОБРАБОТЧИКИ ИНСТРУМЕНТОВ
    # -------------------------------------------------------------------------

    def _handle_find_slot(self, request: ExecuteRequest) -> MLResponse:
        """ML-ранжирование свободных слотов и генерация человечного ответа."""
        slots = request.fetched_slots or []
        ranked = self.ranker.rank_slots(slots, request.context)

        if not ranked:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="К сожалению, Google не нашел свободных слотов в указанном диапазоне.",
                parameters={}
            )

        best_time = ranked[0].start.split("T")[-1][:5]
        alt_time = ranked[1].start.split("T")[-1][:5] if len(ranked) > 1 else "другое"

        prompt = p.SLOT_FOUND_PROMPT.format(
            persona=p.CHAT_PERSONA,
            best_slot_time=best_time,
            alt_slot_time=alt_time
        )

        llm_resp = self._call_model(prompt, request.text)
        reply = llm_resp.get("reply_text",
                             f"Предлагаю время: {best_time}.") if llm_resp else f"Лучшее время: {best_time}"

        return MLResponse(
            tool_name=ToolName.FIND_FREE_SLOT,
            reply_text=reply,
            parameters=RankedSlotParams(
                ranked_slots=ranked[:3],  # Возвращаем топ-3
                reasoning="Selected by AI Ranking (Work hours priority)"
            )
        )

    def _handle_summarize(self, request: ExecuteRequest) -> MLResponse:
        """LLM-саммари событий календаря."""
        events = request.fetched_events or []
        events_str = json.dumps([
            {"title": e.title, "start": e.start} for e in events
        ], ensure_ascii=False)

        prompt = p.SUMMARIZE_PROMPT.format(
            persona=p.CHAT_PERSONA,
            user_text=request.text,
            events_json=events_str
        )

        raw_data = self._call_model(prompt, request.text)
        reply = raw_data.get("reply_text", "События проанализированы.") if raw_data else "Готово."

        return MLResponse(
            tool_name=ToolName.SUMMARIZE_WEEK,
            reply_text=reply,
            parameters={}
        )

    def _handle_update_event(self, request: ExecuteRequest) -> MLResponse:
        """LLM-определение какое событие изменить и как."""
        events = request.fetched_events or []

        if not events:
            return MLResponse(
                tool_name=ToolName.CLARIFICATION_NEEDED,
                reply_text="Не нашёл события для изменения. Уточните, какое событие вы хотите изменить?",
                parameters={}
            )

        events_str = json.dumps([
            {"id": e.id, "title": e.title, "start": e.start, "end": e.end} for e in events
        ], ensure_ascii=False)
        
        prompt = p.UPDATE_EVENT_PROMPT.format(
            persona=p.CHAT_PERSONA,
            user_text=request.text,
            events_json=events_str
        )

        raw_data = self._call_model(prompt, request.text)

        if raw_data:
            reply = raw_data.get("reply_text", "Готово, событие обновлено.")
            params = raw_data.get("parameters", {})
        else:
            reply = "Событие найдено. Что именно нужно изменить?"
            params = {}

        return MLResponse(
            tool_name=ToolName.UPDATE_EVENT,
            reply_text=reply,
            parameters=params
        )

    def _handle_split_task(self, request: ExecuteRequest) -> MLResponse:
        """LLM-разбиение большой задачи на подзадачи с оценкой времени."""
        prompt = p.SPLIT_TASK_PROMPT.format(
            persona=p.CHAT_PERSONA,
            user_text=request.text
        )

        raw_data = self._call_model(prompt, request.text)

        try:
            if not raw_data:
                raise ValueError("Empty LLM response")

            reply = raw_data.get("reply_text", "План готов.")
            params_dict = raw_data.get("parameters", {})
            valid_params = SplitTaskParams(**params_dict)

            return MLResponse(
                tool_name=ToolName.SPLIT_TASK,
                reply_text=reply,
                parameters=valid_params
            )
        except ValidationError:
            logger.warning("split_task bad schema")
            return MLResponse(
                tool_name=ToolName.GENERAL_CHAT,
                reply_text="Я составил план, но возникла ошибка форматирования. Давайте попробуем разбить задачу вручную?",
                parameters={}
            )

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