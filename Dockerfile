# Dockerfile с локальной LLM моделью (llama-cpp-python)
FROM python:3.11-slim-bookworm

WORKDIR /app

# Зависимости для сборки llama-cpp-python и ffmpeg для whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем зависимости
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код (без папки models — она монтируется как volume)
COPY app/ ./app/
COPY requirements.txt .
COPY training_data.csv .

# Создаём папку для модели
RUN mkdir -p /app/models

# Переменные окружения
ENV USE_LOCAL_LLM=true
ENV LOCAL_MODEL_PATH=/app/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]