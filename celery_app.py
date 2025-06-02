from celery import Celery
from celery.schedules import crontab
import os

# Use Redis URL from Heroku if available, otherwise use local
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Initialize Celery app
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,  # Redis as the broker
    backend=REDIS_URL, # Redis as the result backend
    include=["tasks.tasks", "scripts.update_stocks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Add periodic task configuration
celery_app.conf.beat_schedule = {
    "update-tickers": {
        "task": "scripts.update_stocks.update_all_tickers",  # Updated task path
        "schedule": crontab(
            minute='*/15',
            hour='9-16',
            day_of_week='1-5'
        ),
    },
}
