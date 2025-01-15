import os
import dotenv
from openai import OpenAI
import logging

dotenv.load_dotenv()

import logging
import sys

# Create a logger named 'logfile'
logger = logging.getLogger('logfile')

# Set the logger level to DEBUG to capture all levels of logs
logger.setLevel(logging.DEBUG)

# Add a StreamHandler (console) to log at DEBUG level
console_handler = logging.StreamHandler(sys.stdout)  # Use `sys.stdout` for compatibility with Heroku
console_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to console

# Add a FileHandler to log at DEBUG level to a file
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to file

# Define a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Apply the formatter to both console and file handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



def log(msg):
    logger.debug(msg)
    print(msg)


class Config:
    """Base configuration variables."""
    production = False

    # Configure PostgreSQL Database
    username_db = os.getenv('POSTGRES_USER')
    password_db = os.getenv('POSTGRES_PASSWORD')
    database_name = os.getenv('POSTGRES_DB')
    postgres_uri = f'postgresql://{username_db}:{password_db}@localhost/{database_name}'
    EMBEDDINGS_DIR = os.path.join('xyz', 'modules', 'llm', 'embedding_tools', 'embeddings')


class GOOGLE:
    """Google configuration variables."""
    google_search_key = os.getenv('GOOGLE_SEARCH_KEY')
    google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')


class OAI:
    """OpenAI configuration variables."""
    # OpenAI Client
    client = None
    # Models
    gpt4o = "gpt-4o"
    gpt4o_mini = "gpt-4o-mini"
    embedding3 = "text-embedding-3-large"
    dall_e_3 = "dall-e-3"
    tts = "tts-1"
    tts_hd = 'tts-1-hd'
    whisper = "whisper-1"
    moderation = "text-moderation-latest"


openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
OAI.client = openai_client

