import os

import requests
from flask import current_app


class BrowserService:
    def __init__(self):
        self.base_url = os.getenv('BROWSER_SERVICE_URL')
        current_app.config.update(
            BROWSER_SERVICE_URL=self.base_url)

    def start_browser(self):
        try:
            response = requests.post(f"{self.base_url}/browser/start")
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def navigate_to(self, url):
        try:
            response = requests.post(
                f"{self.base_url}/browser/navigate",
                json={"url": url}
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
