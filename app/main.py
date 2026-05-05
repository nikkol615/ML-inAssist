"""
 Модуль: main.py
 Назначение: Главный модуль FastAPI-приложения inAssist ML Backend
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

import asyncio
import time
import logging
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks

from app.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    ExecuteRequest, MLResponse,
    FeedbackRequest, StepRequest, StepResponse
)
from app.services.asr import ASRService
from app.services.llm import LLMService
from app.services.agent import AgentService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_core")

app = FastAPI(title="inAssist ML Backend")

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".opus", ".wav", ".m4a"}
ALLOWED_AUDIO_CONTENT_TYPES = {
    "application/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/x-m4a",
    "audio/x-wav",
    "video/mp4",
}
DEFAULT_MAX_TRANSCRIBE_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

_asr_service = None
_llm_service = None
_agent_service = None


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


def get_agent_service():
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService(get_llm_service())
    return _agent_service


def _max_transcribe_bytes() -> int:
    try:
        return int(os.getenv("MAX_TRANSCRIBE_BYTES", DEFAULT_MAX_TRANSCRIBE_BYTES))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TRANSCRIBE_BYTES


def _collect_readiness_status() -> dict:
    checks = {}

    llm = get_llm_service()
    llm_mode = "local" if llm.use_local else "api"
    if llm.use_local:
        llm_ready = llm.local_model is not None
    else:
        llm_ready = llm.client is not None
    checks["llm"] = {"ready": llm_ready, "mode": llm_mode}

    asr = get_asr_service()
    checks["asr"] = {"ready": asr.model is not None}

    checks["ranking"] = {"ready": llm.ranker is not None}

    ready = all(check["ready"] for check in checks.values())
    return {
        "status": "ok" if ready else "not_ready",
        "ready": ready,
        "checks": checks,
    }


@app.middleware("http")
async def measure_latency(request: Request, call_next):
    """Добавляет заголовок X-Process-Time с временем обработки запроса."""
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start)
    return response


@app.post("/api/v1/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Новый agent-step runtime: один следующий шаг за вызов."""
    return get_agent_service().step(request)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, deprecated=True)
async def analyze(request: AnalyzeRequest):
    """Legacy router endpoint. Оставлен для обратной совместимости."""
    return get_agent_service().legacy_analyze(request)


@app.post("/api/v1/execute", response_model=MLResponse, deprecated=True)
async def execute(request: ExecuteRequest):
    """Legacy execute endpoint. Оставлен для обратной совместимости."""
    return get_agent_service().legacy_execute(request)


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
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp3"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Неподдерживаемый формат аудиофайла")

    content_type = (file.content_type or "").lower()
    if content_type and not (
            content_type.startswith("audio/") or content_type in ALLOWED_AUDIO_CONTENT_TYPES
    ):
        raise HTTPException(status_code=415, detail="Неподдерживаемый MIME-тип аудиофайла")

    tmp_path = None
    try:
        total_size = 0
        max_size = _max_transcribe_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(status_code=413, detail="Аудиофайл слишком большой")
                tmp.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Пустой аудиофайл")

        text = await asyncio.to_thread(lambda: get_asr_service().transcribe(tmp_path))
        if not text:
            raise HTTPException(status_code=500, detail="Не удалось распознать речь")
        return {"text": text}
    finally:
        await file.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
async def health():
    """Readiness check для Docker/Kubernetes."""
    status = await asyncio.to_thread(_collect_readiness_status)
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status)
    return status


@app.post("/api/v1/retrain")
async def retrain_ranking_model():
    """Принудительно переобучает LightGBM-ранкер на накопленных данных."""
    try:
        ranker = await asyncio.to_thread(lambda: get_llm_service().ranker)
        await asyncio.to_thread(ranker._maybe_retrain)
        return {
            "status": "ok",
            "model_active": ranker.use_ml,
            "message": "Model retrained successfully" if ranker.use_ml else "Not enough data yet"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model-status")
async def model_status():
    """Возвращает статус ML-ранкера: активен ли, сколько примеров собрано."""
    sample_count = 0
    data_file = "/app/data/training_data.csv" if os.path.exists("/app/data") else "training_data.csv"
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            sample_count = sum(1 for _ in f) - 1
    except (FileNotFoundError, IOError):
        pass

    return {
        "ranking_model_active": get_llm_service().ranker.use_ml,
        "training_samples": sample_count,
        "min_samples_required": 20,
        "llm_mode": "local" if os.getenv("USE_LOCAL_LLM", "false").lower() == "true" else "api"
    }
