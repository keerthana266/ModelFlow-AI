FROM python:3.12-slim

WORKDIR /runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/models/model.joblib

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY app ./app

RUN useradd --create-home --uid 10001 modeluser \
    && mkdir -p /models \
    && chown -R modeluser:modeluser /runtime /models

USER modeluser
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
