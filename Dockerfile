FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# La API necesita el código, los artefactos entrenados y los CSV históricos.
COPY src ./src
COPY models ./models
COPY F1 ./F1
COPY UFC ./UFC

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn f1_ranker.api:app --host 0.0.0.0 --port ${PORT}"]
