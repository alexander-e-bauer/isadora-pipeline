
import os
import time
import json
import ast
import sys
import ssl
import urllib3
from cachetools import TTLCache
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_security import Security, SQLAlchemyUserDatastore, UserMixin, RoleMixin
from flask_security import RegisterForm, LoginForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from flask_wtf.csrf import CSRFProtect
from gunicorn.sock import ssl_context
import eventlet


eventlet.monkey_patch(thread=False)  # Add thread=False to help with recursion issues
sys.setrecursionlimit(3000)

import numpy as np
import pandas as pd
import redis
from keras import Model, Sequential
from keras.src.saving import load_model
import tensorflow as tf

from xyz.modules.gateway import gateway_blueprint
from xyz.modules.llm import llm_blueprint, embedding_tool
from xyz.modules.database import database
import config
OAI = config.OAI
logger = config.logger


# Increase recursion limit and configure SSL
sys.setrecursionlimit(3000)

# SSL Configuration
def create_ssl_context():
    try:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    except Exception as e:
        logger.error(f"Failed to create SSL context: {e}")
        return None

# Configure SSL for requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl_context = create_ssl_context()

# In-memory storage for conversation history
conversation_history = {}

app = Flask(__name__)
# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",  # React development server
            "http://localhost:5000",  # Flask development server
            "http://localhost:6379",  # Flask development server
            "https://chat-widget-app-8c3cca0ff3c0.herokuapp.com",  # Production URL
            "https://alexander-e-bauer.github.io"  # Add your frontend domain
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


# Configure SocketIO with CORS settings
socketio = SocketIO(app, cors_allowed_origins=[
    "http://localhost:3000",
    "http://localhost:5000",
    "https://chat-widget-app-8c3cca0ff3c0.herokuapp.com",
    "https://alexander-e-bauer.github.io"], async_mode='eventlet'
)

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")

username_db = os.getenv('POSTGRES_USER')
password_db = os.getenv('POSTGRES_PASSWORD')
database_name = os.getenv('POSTGRES_DB')
postgres_uri = f'postgresql://{username_db}:{password_db}@localhost/{database_name}'
app.config['SQLALCHEMY_DATABASE_URI'] = postgres_uri

app.config['SECURITY_PASSWORD_SALT'] = os.environ.get("SECURITY_PASSWORD_SALT")

# Create database connection object
db = database.database.init_app(app)

# Define models
User, Role, roles_users = database.models.define_models(db)

# Setup Flask-Security
user_datastore = SQLAlchemyUserDatastore(db, User, Role)
app.config['SECURITY_REGISTERABLE'] = True
app.config['SECURITY_RECOVERABLE'] = True
app.config['SECURITY_CHANGEABLE'] = True
app.config['SECURITY_REGISTER_URL'] = '/register'
app.config['SECURITY_LOGIN_URL'] = '/login'
app.config['SECURITY_LOGOUT_URL'] = '/logout'
app.config['SECURITY_RESET_URL'] = '/reset'
app.config['SECURITY_CHANGE_URL'] = '/change'

class ExtendedRegisterForm(RegisterForm):
    first_name = StringField('First Name', [DataRequired()])
    last_name = StringField('Last Name', [DataRequired()])

class ExtendedLoginForm(LoginForm):
    email = StringField('Email Address', [DataRequired()])

security = Security(app, user_datastore,
                    register_form=ExtendedRegisterForm,
                    login_form=ExtendedLoginForm)

# CSRF
csrf = CSRFProtect(app)

# Bootstrap
Bootstrap(app)

# Blueprints
gw = gateway_blueprint.init_app(app)
oai = llm_blueprint.init_app(app)


@socketio.on('typing')
def handle_typing(data):
    emit('typing', data, broadcast=True, include_self=False)

@socketio.on('stop_typing')
def handle_stop_typing(data):
    emit('stop_typing', data, broadcast=True, include_self=False)


@lru_cache(maxsize=100)
def load_keras_model(ticker):
    """Load and cache the Keras model in memory."""
    try:
        model_path = f"models/{ticker}_model.h5"
        if os.path.exists(model_path):
            return load_model(model_path)
        logger.warning(f"No model found for {ticker}")
        return None
    except Exception as e:
        logger.error(f"Error loading model for {ticker}: {str(e)}")
        return None


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    return embedding_tool.jsonify_chat(data, conversation_history)


@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"


if __name__ == '__main__':
    if config.Config.production:
        socketio.run(app,
                    host='0.0.0.0',
                    port=5000,
                    debug=False,  # Set to False in production
                    ssl_context=ssl_context,
                    allow_unsafe_werkzeug=True)
    else:
        socketio.run(app,
                    host='0.0.0.0',
                    port=5000,
                    debug=True,
                    ssl_context=ssl_context,
                    allow_unsafe_werkzeug=True)