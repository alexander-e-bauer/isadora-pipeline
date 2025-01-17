# browser_service.py
import os
import requests
from flask import current_app

import config
logger = config.logger
from threading import Lock

lock = Lock()

class BrowserService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.initialized = False
        self.base_url = None

    def initialize_with_app(self, app):
        """
        Initialize the service with Flask app configuration
        """
        if not self.initialized:
            app.config.update(
                BROWSER_SERVICE_URL=os.getenv('BROWSER_SERVICE_URL', 'https://isadora.ai')
            )
            self.base_url = app.config['BROWSER_SERVICE_URL']
            self.initialized = True

    def start_browser(self):
        """
        Interacts with the /api/browser/start endpoint at the backend
        """
        try:
            response = requests.post(f"{self.base_url}/api/browser/start")
            response.raise_for_status()
            current_app.logger.debug(
                f"5a: Start Request posted to VM\nResponse: {response}")
            return response.json()
        except requests.RequestException as e:
            current_app.logger.error(f"Failed to start browser: {e}")
            return {"status": "error", "message": str(e)}

    def navigate_to_url(self, url):
        """
        Interacts with the /api/browser/navigate endpoint to navigate to a URL
        """
        try:
            data = {"url": url}
            response = requests.post(f"{self.base_url}/api/browser/navigate", json=data)
            logger.info(f"5b: Request posted to VM\nResponse: {response.content}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            current_app.logger.error(f"Failed to navigate to URL {url}: {e}")
            logger.error(f"5b: Failed to start browser: {e}")
            return {"status": "error", "message": str(e)}

    def get_page_content(self):
        """
        Interacts with the /api/browser/content endpoint to retrieve webpage content.
        """
        with lock:
            try:
                response = requests.get(f"{self.base_url}/api/browser/content")
                current_app.logger.debug(
                    f"Fetching Web Content Request posted to VM\nResponse: {response.content}")
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                current_app.logger.error(f"Failed to get page content: {e}")
                return {"status": "error", "message": str(e)}

    def check_status(self):
        """
        Interacts with the /api/browser/status endpoint to get browser status
        """
        try:
            response = requests.get(f"{self.base_url}/api/browser/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            current_app.logger.error(f"Failed to get browser status: {e}")
            return {"status": "error", "message": str(e)}
