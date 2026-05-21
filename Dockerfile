FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_DEBUG=0
ENV LOG_LEVEL=INFO
ENV HOST=0.0.0.0
ENV PREDICTOR_DEMO_MODE=true

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "run_webapp.py"]
