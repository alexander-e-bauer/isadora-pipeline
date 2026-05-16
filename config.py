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
console_handler = logging.StreamHandler()#sys.stdout)  # Use `sys.stdout` for compatibility with Heroku
console_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to console

# Define a log format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Apply the formatter to both console and file handlers
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)

key = os.getenv('KEY')

def log(msg):
    logger.debug(msg)
    print(msg)

class DATABASE:
    DB_HOST = os.getenv('DB_HOST')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_NAME = os.getenv('DB_NAME')



class AIS:
    # List of known bad user-agent patterns (bots to block)
    BLOCKED_USER_AGENTS = [
        r'curl',  # Block basic curl requests
        r'wget',  # Block wget requests
        r'python-requests',  # Block Python requests
        r'httpclient',  # Block generic HTTP clients
        r'libwww-perl',  # Block Perl-based HTTP clients
        r'bot',  # Generic bot keyword
        r'scrapy',  # Block Scrapy spiders
        r'java',  # Block Java-based HTTP clients
    ]
    # List of suspicious patterns to block
    SUSPICIOUS_PATTERNS = [
        r'\.env',  # Block .env file requests
        r'\.git',  # Block .git folder or related requests
        r'\.htaccess',  # Block .htaccess file requests
        r'wp-config\.php',  # Block WordPress config file requests
        r'phpinfo\.php',  # Block requests for phpinfo.php
        r'composer\.json',  # Block composer.json requests
        r'/etc/passwd',  # Block attempts to access Linux password file
        r'adminer\.php',  # Block requests for adminer.php
        r'wp-admin',  # Block requests for WordPress admin paths
        r'config\.yml',  # Block YAML configuration files
        r'webui',  # Block requests for web interfaces
        r'geoserver',  # Block requests for geospatial servers
        r'login\.php',  # Block login attempts to PHP-based systems
        r'xmlrpc\.php',  # Common WordPress attack vector
        r'cgi-bin',  # Block CGI-bin folder access
    ]
    MAX_REQUESTS_PER_MINUTE = 60
    BLOCKED_COUNTRIES = []
    HONEYPOT_ENDPOINTS = []
    IPINFO_TOKEN = os.getenv('IPINFO_TOKEN')


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
    gpt4o = "gpt-4.1"
    gpt4o_mini = "o4-mini"
    embedding3 = "text-embedding-3-small"
    dall_e_3 = "dall-e-3"
    tts = "gpt-4o-mini-tts"
    tts_hd = 'tts-1-hd'
    whisper = "whisper-1"
    moderation = "text-moderation-latest"


openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
OAI.client = openai_client

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_HOST = os.getenv('PINECONE_HOST')
SK = os.getenv('SKELETON_KEY')
FINAZON_KEY = os.getenv('FINAZON_KEY')
POLYGON_KEY = os.getenv('POLYGON_KEY')
AV_KEY = os.getenv('AV_KEY')