# Dockerfile с локальной LLM моделью (llama-cpp-python)
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update && apt-get install -y --fix-missing --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY app/ ./app/
COPY requirements.txt .


RUN mkdir -p /app/models /app/data


COPY training_data.csv /app/data/training_data.csv


ENV USE_LOCAL_LLM=true
ENV LOCAL_MODEL_PATH=/app/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
ENV PYTHONUNBUFFERED=1

EXPOSE 3000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000", "--log-level", "info"]
