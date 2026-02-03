# ML Service

Бэкенд для обработки голоса и текста через LLM.

## Подготовка

Для локального режима необходимо скачать веса в папку `models/`:
- `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` — [Hugging Face](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF)


## Переменные окружения

```
GROQ_API_KEY=gsk_...        # API ключ Groq
USE_LOCAL_LLM=false         # true = локальная модель, false = Groq API
LOCAL_MODEL_PATH=/app/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

## API

### POST /api/v1/analyze
Анализ текста пользователя, определение намерения.

### POST /api/v1/execute  
Выполнение действия с данными от Google Calendar.

### POST /api/v1/transcribe
Распознавание речи (Whisper).

### POST /api/v1/feedback
Сохранение фидбека для обучения ранжирования.

### GET /health
Проверка статуса.

## Структура

```
ml_service/
├── Dockerfile              # Сборка Docker-образа
├── docker-compose.yml      # Конфигурация контейнера
├── requirements.txt        # Python зависимости
├── training_data.csv       # Данные для обучения ML-ранкера
├── models/
│   └── .gitkeep            # (модель .gguf скачивается отдельно)
└── app/
    ├── __init__.py
    ├── main.py             # FastAPI endpoints
    ├── schemas.py          # Pydantic модели
    ├── prompts.py          # Промпты для LLM
    └── services/
        ├── __init__.py
        ├── asr.py          # Whisper (распознавание речи)
        ├── llm.py          # Groq/Llama (генерация)
        └── ranking.py      # LightGBM (ранжирование слотов)
```
