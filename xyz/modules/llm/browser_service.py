# browser_service.py
import os

from flask import current_app


class BrowserService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Move the config update to a separate method
        self.initialized = False

    def initialize_with_app(self, app):
        if not self.initialized:
            app.config.update(
                BROWSER_SERVICE_URL=os.getenv('BROWSER_SERVICE_URL')
            )
            self.initialized = True
