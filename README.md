# ML Service

Backend для календарного ассистента `inAssist`: ML-сервис для обработки текстовых и голосовых команд, выбора следующего действия агента и формирования JSON-ответов для Gateway.

Главный runtime теперь построен вокруг agent-step схемы:

1. gateway отправляет пользовательское сообщение и текущее состояние сессии в `/api/v1/step`
2. LLM возвращает ровно один следующий шаг:
   - вызвать tool
   - уточнить запрос
   - завершить задачу
3. gateway выполняет tool, складывает observation в state и снова вызывает `/api/v1/step`
4. цикл повторяется, пока задача не завершится

Старые `/api/v1/analyze` и `/api/v1/execute` оставлены только для обратной совместимости.
Gateway после каждого шага сам применяет `state_patch`, добавляет `completed_actions` и `tool_observations`, а затем снова вызывает `/api/v1/step`.

## Подготовка

Для локального режима нужно скачать веса в `models/`:

- `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` — [Hugging Face](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)

## Переменные окружения

```env
GROQ_API_KEY=gsk_...
USE_LOCAL_LLM=false
LOCAL_MODEL_PATH=/app/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
MAX_TRANSCRIBE_BYTES=26214400
```

## Основной API

### POST `/api/v1/step`

Главная agent-step ручка. Gateway передаёт текст, временной контекст и состояние сценария; сервис возвращает ровно один следующий шаг.

Формат запроса:

- `text` — новое сообщение пользователя или текущая инструкция шага
- `context` — `current_time`, `timezone`, рабочие часы пользователя
- `state` — структурированное состояние сценария
- `conversation` — legacy-контекст, если полноценный `state` ещё не передан
- `all_context` — запасной сырой контекст, используется только как fallback

Минимальный запрос:

```json
{
  "text": "поменяй местами события стоматолога и футбол",
  "context": {
    "current_time": "2026-03-25T18:00:00+03:00",
    "timezone": "Europe/Moscow",
    "work_start_hour": 9,
    "work_end_hour": 18
  },
  "state": {
    "messages": [
      {
        "role": "user",
        "text": "поменяй местами события стоматолога и футбол"
      }
    ],
    "completed_actions": [],
    "tool_observations": [],
    "working_state": {
      "status": "in_progress"
    }
  }
}
```

Типовой ответ с tool call:

```json
{
  "response_type": "tool_call",
  "assistant_message": null,
  "response_payload": {},
  "next_gateway_action": {
    "type": "tool_call",
    "tool_call": {
      "tool_name": "find_event",
      "arguments": {
        "query": "стоматолог",
        "max_results": 10
      }
    }
  },
  "state_patch": {
    "working_state": {
      "goal": "swap_two_events",
      "status": "in_progress",
      "plan_summary": "find dentist event, find football event, swap their times"
    }
  }
}
```

Типовой ответ с уточнением:

```json
{
  "response_type": "clarify",
  "assistant_message": "Уточни, пожалуйста, какое именно событие нужно изменить.",
  "response_payload": {},
  "next_gateway_action": {
    "type": "none",
    "tool_call": null
  },
  "state_patch": {
    "working_state": {
      "status": "awaiting_user"
    }
  }
}
```

Типовой финальный ответ:

```json
{
  "response_type": "finish",
  "assistant_message": "Готово, я поменял местами эти события.",
  "response_payload": {},
  "next_gateway_action": {
    "type": "none",
    "tool_call": null
  },
  "state_patch": {
    "working_state": {
      "status": "done"
    }
  }
}
```

### Session state

Основные отсеки state:

- `messages` — история диалога user/assistant
- `completed_actions` — уже выполненные gateway actions
- `tool_observations` — реальные результаты tools
- `working_state` — текущая цель, статус, pending action и resolved entities
- `memory_summary` — короткое резюме старого контекста, если история уже схлопнута

Формат ответа:

- `response_type` — `tool_call`, `clarify` или `finish`
- `assistant_message` — сообщение пользователю для `clarify`/`finish`
- `response_payload` — дополнительные структурированные данные, если нужны
- `next_gateway_action` — следующий tool call для Gateway или `none`
- `state_patch` — частичное обновление `working_state`

### Доступные gateway tools

- `find_event` — поиск событий по приблизительному названию или query
- `list_events` — получение событий за период
- `get_free_slots` — получение свободных окон
- `create_event` — создание события
- `update_event` — изменение события по `event_id`
- `delete_event` — удаление события по `event_id`

## Legacy API

### POST `/api/v1/analyze`

Старый router endpoint. Оставлен как deprecated-совместимость.

### POST `/api/v1/execute`

Старый execute endpoint. Оставлен как deprecated-совместимость.

## Остальные endpoints

### POST `/api/v1/transcribe`

Распознавание речи через faster-whisper. Поддерживаются аудиофайлы `mp3`, `ogg`, `opus`, `wav`, `m4a`; ограничение размера задаётся через `MAX_TRANSCRIBE_BYTES`.

### POST `/api/v1/feedback`

Сохранение пользовательского выбора слота для обучения ранкера.

### POST `/api/v1/retrain`

Ручной запуск переобучения ранкера.

### GET `/api/v1/model-status`

Статус ранкера и режим LLM.

### GET `/health`

Healthcheck.

## Структура

```text
ML-inAssist-main/
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- training_data.csv
|-- models/
`-- app/
    |-- __init__.py
    |-- main.py
    |-- prompts.py
    |-- schemas.py
    `-- services/
        |-- __init__.py
        |-- agent.py
        |-- asr.py
        |-- llm.py
        `-- ranking.py
```
