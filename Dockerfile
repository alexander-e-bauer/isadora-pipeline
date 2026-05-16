FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run expects your app to listen on $PORT (default 8080)
ENV PORT=8080

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app