from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading
import queue
import os


class BrowserController:
    def __init__(self):
        self.driver = None
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.browser_thread = None

    def start_browser(self):
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")

            service = Service(os.environ.get("CHROMEDRIVER_PATH"))
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            return {"status": "success", "message": "Browser started"}

    def navigate_to(self, url):
        if not self.driver:
            return {"status": "error", "message": "Browser not started"}
        try:
            self.driver.get(url)
            return {
                "status": "success",
                "title": self.driver.title,
                "url": self.driver.current_url
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _browser_worker(self):
        while self.is_running:
            try:
                command, args = self.command_queue.get(timeout=1)
                result = getattr(self, command)(*args)
                self.result_queue.put(("success", result))
            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put(("error", str(e)))


    def get_page_content(self):
        return {
            "title": self.driver.title,
            "url": self.driver.current_url,
            "content": self.driver.page_source
        }

    def execute_command(self, command, *args):
        self.command_queue.put((command, args))
        status, result = self.result_queue.get()
        if status == "error":
            raise Exception(result)
        return result

    def cleanup(self):
        self.is_running = False
        if self.driver:
            self.driver.quit()
            self.driver = None

def bitch():
    fluffin = "George"
