"""
 Модуль: asr.py
 Назначение: Сервис распознавания речи (ASR) на базе Whisper
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

import os
import logging
import threading

from faster_whisper import WhisperModel

logger = logging.getLogger("ml_asr")
logger.setLevel(logging.INFO)


class ASRService:
    """Faster-Whisper ASR: аудио → текст (base model, INT8, CPU)."""

    def __init__(self):
        model_size = "base"
        device = "cpu"
        compute_type = "int8"
        self._lock = threading.Lock()

        logger.info(f"loading whisper {model_size}")

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4
            )
            logger.info("asr ready")
        except Exception as e:
            logger.critical(f"whisper failed: {e}")
            self.model = None

    def transcribe(self, file_path: str) -> str:
        """mp3/ogg/wav/m4a → текст. Требует ffmpeg."""
        if not self.model:
            logger.error("asr not init")
            return ""

        if not os.path.exists(file_path):
            logger.error(f"file not found: {file_path}")
            return ""

        logger.info(f"transcribe: {file_path}")

        try:
            with self._lock:
                segments, info = self.model.transcribe(
                    file_path,
                    beam_size=5,
                    language="ru",
                    vad_filter=True
                )

                full_text = " ".join(segment.text for segment in segments).strip()

            logger.info(f"done {info.duration:.1f}s: {full_text[:50]}")
            return full_text

        except Exception as e:
            logger.error(f"transcribe err: {e}")
            return ""
