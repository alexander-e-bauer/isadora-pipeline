import os
import sys
import ssl
import urllib3

from flask import Flask, request, make_response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_security import Security, SQLAlchemyUserDatastore
from flask_security import RegisterForm, LoginForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap
from flask_wtf.csrf import CSRFProtect
import eventlet

eventlet.monkey_patch(thread=False)
sys.setrecursionlimit(3000)

from xyz.modules.llm import llm_blueprint, embedding_tool
from xyz.modules.database import database, models
import config

OAI = config.OAI
logger = config.logger


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

# CORS Configuration with all necessary headers
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://isadora-v2-74e5a1b97f07.herokuapp.com",
            "https://isadora-f5fbebf38bc6.herokuapp.com"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Accept",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers"
        ],
        "expose_headers": [
            "Content-Type",
            "Authorization",
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Credentials"
        ],
        "supports_credentials": True,
        "max_age": 86400  # Cache preflight requests for 24 hours
    }
})

# Updated SocketIO Configuration
socketio = SocketIO(
    app,
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "https://isadora-v2-74e5a1b97f07.herokuapp.com",
        "https://isadora-f5fbebf38bc6.herokuapp.com"
    ],
    async_mode='eventlet',
    ping_timeout=30,
    ping_interval=15,
    always_connect=True,
    path='/socket.io',
    transport=['websocket', 'polling'],
    cookie=False
)


# ... (keep all your existing configurations)

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in [
        "http://localhost:3000",
        "http://localhost:5000",
        "https://isadora-v2-74e5a1b97f07.herokuapp.com",
        "https://isadora-f5fbebf38bc6.herokuapp.com"
    ]:
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    return response


# Updated Routes with proper CORS handling
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = make_response()
        origin = request.headers.get('Origin')
        if origin in [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://isadora-v2-74e5a1b97f07.herokuapp.com",
            "https://isadora-f5fbebf38bc6.herokuapp.com"
        ]:
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers[
            'Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Max-Age'] = '86400'
        return response

    data = request.json
    result = embedding_tool.jsonify_chat(data, conversation_history)
    response = make_response(result)
    return response


@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    if request.method == 'OPTIONS':
        response = make_response()
    else:
        response = make_response("Hello, World!")

    origin = request.headers.get('Origin')
    if origin in [
        "http://localhost:3000",
        "http://localhost:5000",
        "https://isadora-v2-74e5a1b97f07.herokuapp.com",
        "https://isadora-f5fbebf38bc6.herokuapp.com"
    ]:
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response


# Error handling with CORS headers
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"An error occurred: {error}")
    response = jsonify({
        "error": str(error),
        "message": "An internal error occurred"
    })
    origin = request.headers.get('Origin')
    if origin in [
        "http://localhost:3000",
        "http://localhost:5000",
        "https://isadora-v2-74e5a1b97f07.herokuapp.com",
        "https://isadora-f5fbebf38bc6.herokuapp.com"
    ]:
        response.headers['Access-Control-Allow-Origin'] = origin
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response, 500



# Debug logging
@app.before_request
def log_request_info():
    logger.debug('Headers: %s', request.headers)
    logger.debug('Body: %s', request.get_data())

@app.after_request
def after_request(response):
    logger.debug('Response Headers: %s', response.headers)
    return response

# Socket.IO event handlers
@socketio.on('typing')
def handle_typing(data):
    emit('typing', data, broadcast=True, include_self=False)

@socketio.on('stop_typing')
def handle_stop_typing(data):
    emit('stop_typing', data, broadcast=True, include_self=False)

@socketio.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {e}")
    return {"error": str(e)}

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"SocketIO default error: {e}")
    return {"error": str(e)}

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

# Updated Routes with CORS handling
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response

    data = request.json
    result = embedding_tool.jsonify_chat(data, conversation_history)
    response = make_response(result)
    response.headers.add('Access-Control-Allow-Origin', request.headers.get('Origin', '*'))
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/', methods=['GET'])
def index():
    response = make_response("Hello, World!")
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Error handling
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"An error occurred: {error}")
    response = jsonify({
        "error": str(error),
        "message": "An internal error occurred"
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 500

if __name__ == '__main__':
    if config.Config.production:
        socketio.run(
            app,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=False,
            use_reloader=False,
            cors_allowed_origins='*',
            allow_unsafe_werkzeug=True
        )
    else:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            ssl_context=ssl_context,
            cors_allowed_origins='*'
        )
