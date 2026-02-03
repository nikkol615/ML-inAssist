"""
 Модуль: main.py
 Назначение: Главный модуль FastAPI-приложения inAssist ML Backend
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════════════════════

import time
import logging
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks

from app.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    ExecuteRequest, MLResponse,
    FeedbackRequest
)
from app.services.asr import ASRService
from app.services.llm import LLMService


# ═══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_core")

app = FastAPI(title="inAssist ML Backend")


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY INITIALIZATION СЕРВИСОВ
# ═══════════════════════════════════════════════════════════════════════════════

_asr_service = None
_llm_service = None


def get_asr_service():
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def measure_latency(request: Request, call_next):
    """Добавляет заголовок X-Process-Time с временем обработки запроса."""
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS: ОСНОВНЫЕ
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Классифицирует намерение пользователя и определяет нужный инструмент."""
    return get_llm_service().analyze(request)


@app.post("/api/v1/execute", response_model=MLResponse)
async def execute(request: ExecuteRequest):
    """Обрабатывает данные от Google Calendar и формирует ответ."""
    return get_llm_service().execute(request)


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS: ФИДБЕК И ТРАНСКРИПЦИЯ
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/feedback")
async def feedback(req: FeedbackRequest, background_tasks: BackgroundTasks):
    """Принимает фидбек о выбранном слоте для обучения ML-ранкера."""
    background_tasks.add_task(
        get_llm_service().ranker.save_training_data,
        req.context,
        req.chosen_slot,
        req.rejected_slots
    )
    return {"status": "accepted"}

@app.post("/api/v1/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Конвертирует аудиофайл (mp3/ogg/wav/m4a) в текст через Whisper."""
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        text = get_asr_service().transcribe(tmp_path)
        if not text:
            raise HTTPException(status_code=500, detail="Не удалось распознать речь")
        return {"text": text}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS: СЛУЖЕБНЫЕ
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Healthcheck для Docker/Kubernetes."""
    return {"status": "ok"}


@app.post("/api/v1/retrain")
async def retrain_ranking_model():
    """Принудительно переобучает LightGBM-ранкер на накопленных данных."""
    try:
        get_llm_service().ranker._maybe_retrain()
        return {
            "status": "ok",
            "model_active": get_llm_service().ranker.use_ml,
            "message": "Model retrained successfully" if get_llm_service().ranker.use_ml else "Not enough data yet"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/model-status")
async def model_status():
    """Возвращает статус ML-ранкера: активен ли, сколько примеров собрано."""
    sample_count = 0
    try:
        with open("training_data.csv", "r") as f:
            sample_count = sum(1 for _ in f) - 1  # Минус заголовок
    except (FileNotFoundError, IOError):
        pass
    
    return {
        "ranking_model_active": get_llm_service().ranker.use_ml,
        "training_samples": sample_count,
        "min_samples_required": 20,
        "llm_mode": "local" if os.getenv("USE_LOCAL_LLM", "false").lower() == "true" else "api"
    }

