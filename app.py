import os
import sys
import ssl
import urllib3
from flask import Flask, request, make_response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet

from xyz.modules.llm.embedding_tools.embedding_model import read_embedding

eventlet.monkey_patch(thread=False)
sys.setrecursionlimit(3000)

from xyz.modules.llm import embedding_tool
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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl_context = create_ssl_context()

conversation_history = {}

app = Flask(__name__)

# Explicitly define allowed origins
ALLOWED_ORIGINS = [
    'https://isadora-f5fbebf38bc6.herokuapp.com',
    'https://isadora-v2-74e5a1b97f07.herokuapp.com',
    'http://localhost:3000'
]

# Updated CORS configuration with explicit options
CORS(app,
     resources={
         r"/*": {
             "origins": ALLOWED_ORIGINS,
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
                 "Authorization"
             ],
             "supports_credentials": True,
             "send_wildcard": False,
             "max_age": 86400
         }
     })

# Configure SocketIO with explicit CORS settings
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode='eventlet',
    ping_timeout=60000,
    ping_interval=25000,
    always_connect=True,
    path='/socket.io',
    transport=['websocket', 'polling'],
    cookie=False,
    cors_credentials=True
)

# Flask configuration
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY"),
    SQLALCHEMY_DATABASE_URI=f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost/{os.getenv("POSTGRES_DB")}',
    SECURITY_PASSWORD_SALT=os.environ.get("SECURITY_PASSWORD_SALT"),
    # Security configuration
    SECURITY_REGISTERABLE=True,
    SECURITY_RECOVERABLE=True,
    SECURITY_CHANGEABLE=True,
    SECURITY_REGISTER_URL='/register',
    SECURITY_LOGIN_URL='/login',
    SECURITY_LOGOUT_URL='/logout',
    SECURITY_RESET_URL='/reset',
    SECURITY_CHANGE_URL='/change',
    # CORS settings
    SECURITY_CSRF_COOKIE_NAME="XSRF-TOKEN",
    SECURITY_CSRF_HEADER_NAME="X-XSRF-TOKEN",
    WTF_CSRF_CHECK_DEFAULT=False,  # Disable CSRF for API endpoints
    WTF_CSRF_TIME_LIMIT=None
)


df = embedding_tool.read_code()
#df = embedding_tool.read_directory('xyz/modules/llm/embedding_tools/embeddings/source_documents', 'source_documents', update=True)

# Debug logging
@app.before_request
def log_request_info():
    logger.debug('Headers: %s', request.headers)
    logger.debug('Body: %s', request.get_data())

# Explicit CORS headers middleware
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers.update({
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Accept, Origin',
            'Access-Control-Max-Age': '86400'
        })
    logger.debug('Response Headers: %s', response.headers)
    return response

# Explicit OPTIONS handler for preflight requests
@app.route('/api/chat', methods=['OPTIONS'])
def handle_preflight():
    response = make_response()
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers.update({
            'Access-Control-Allow-Origin': origin,
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Accept, Origin',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Max-Age': '86400'
        })
    return response


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        result = embedding_tool.jsonify_chat(data, conversation_history=conversation_history, df=df)
        response = make_response(result)
        origin = request.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            response.headers.update({
                'Access-Control-Allow-Origin': origin,
                'Access-Control-Allow-Credentials': 'true'
            })
        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        error_response = jsonify({
            "error": str(e),
            "message": "Failed to process chat request"
        })
        error_response.status_code = 500
        return error_response

# Socket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connect', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect(data=None):
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('typing')
def handle_typing(data):
    try:
        emit('typing', data, broadcast=True, include_self=False)
    except Exception as e:
        logger.error(f"Error in typing event: {e}")
        emit('error', {'message': 'Failed to broadcast typing status'})

@socketio.on('stop_typing')
def handle_stop_typing(data):
    try:
        emit('stop_typing', data, broadcast=True, include_self=False)
    except Exception as e:
        logger.error(f"Error in stop_typing event: {e}")
        emit('error', {'message': 'Failed to broadcast stop typing status'})

@socketio.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {e}")
    return {"error": str(e)}

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"SocketIO default error: {e}")
    return {"error": str(e)}

# Error handling
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"An error occurred: {error}")
    response = jsonify({
        "error": str(error),
        "message": "An internal error occurred"
    })
    return response, 500


@app.route('/update_embeddings')
def update_embeddings():
    global df
    df = embedding_tool.read_directory('xyz/modules/llm/embedding_tools/embeddings/source_documents',
                                       'source_documents', update=True)


if __name__ == '__main__':
    if config.Config.production:
        socketio.run(
            app,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=False,
            use_reloader=False,
            cors_allowed_origins=ALLOWED_ORIGINS,
            allow_unsafe_werkzeug=True
        )
    else:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            ssl_context=ssl_context,
            cors_allowed_origins=ALLOWED_ORIGINS
        )