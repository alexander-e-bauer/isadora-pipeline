web: gunicorn app:app
worker: celery -A celery_app.celery_app worker --concurrency=1 --loglevel=info
beat: celery -A celery_app.celery_app beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler