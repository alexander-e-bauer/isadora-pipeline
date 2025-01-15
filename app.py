import os
import sys
import tempfile
import ssl
import urllib3
from flask import Flask, request, make_response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from gevent import monkey

monkey.patch_all()
from werkzeug.serving import WSGIRequestHandler

from xyz.modules.llm.llm_blueprint import init_app as init_llm_bp
from xyz.modules.llm.browser_service import BrowserService

sys.setrecursionlimit(3000)

from xyz.modules.llm import embedding_tool
import config

OAI = config.OAI
logger = config.logger

x = 100

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Use temp directory or environment variable
EMBEDDINGS_DIR = os.getenv('EMBEDDINGS_DIR', os.path.join(tempfile.gettempdir(), 'embeddings'))

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

WSGIRequestHandler.timeout = 600

# Explicitly define allowed origins
ALLOWED_ORIGINS = [
    'https://isadora-f5fbebf38bc6.herokuapp.com',
    'https://isadora-v2-74e5a1b97f07.herokuapp.com',
    'https://34.16.120.105',
    'http://34.16.120.105',  # Add HTTP version
    'https://isadora.ai',
    'http://localhost:3000',
    '*'  # During development/testing only
]


# Updated CORS configuration with explicit options
# Replace the complex CORS configuration with a simpler one
CORS(app,
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With",
                   "Accept", "Origin", "Access-Control-Request-Method",
                   "Access-Control-Request-Headers"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


# Configure SocketIO with explicit CORS settings
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode='gevent',
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
    SECURITY_REGISTERABLE=True,
    SECURITY_RECOVERABLE=True,
    SECURITY_CHANGEABLE=True,
    SECURITY_REGISTER_URL='/register',
    SECURITY_LOGIN_URL='/login',
    SECURITY_LOGOUT_URL='/logout',
    SECURITY_RESET_URL='/reset',
    SECURITY_CHANGE_URL='/change',
    SECURITY_CSRF_COOKIE_NAME="XSRF-TOKEN",
    SECURITY_CSRF_HEADER_NAME="X-XSRF-TOKEN",
    WTF_CSRF_CHECK_DEFAULT=False,
    WTF_CSRF_TIME_LIMIT=None,
    BROWSER_SERVICE_URL=os.getenv('BROWSER_SERVICE_URL'),
    BROWSER_SERVICE_API_KEY=os.getenv('BROWSER_SERVICE_API_KEY')
)

with app.app_context():
    browser_service = BrowserService.get_instance()
    browser_service.initialize_with_app(app)

df = embedding_tool.initialize_code_embeddings()


@app.before_request
def log_request_info():
    logger.debug('Headers: %s', request.headers)
    logger.debug('Body: %s', request.get_data())


@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS or '*' in ALLOWED_ORIGINS:
        response.headers.update({
            'Access-Control-Allow-Origin': origin or '*',
            'Access-Control-Allow-Credentials': 'true',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With, Accept, Origin',
            'Access-Control-Expose-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400'
        })
    return response


@app.route('/api/browser/navigate', methods=['POST'])
def navigate():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"status": "error", "message": "URL is required"}), 400

    try:
        result = browser_service.navigate_to(url)
        logger.debug(f"Navigated to {url} with result: {result}")

        # Emit the result through socket.io
        socketio.emit('window_update', {
            'content': result,
            'mode': 'browser'
        })

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        logger.error(f"Navigation error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


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
        logger.info("Received chat request")
        data = request.json
        logger.debug(f"Request data: {data}")

        logger.info("Processing chat request with embedding_tool")
        result = embedding_tool.process_chat_request(data, conversation_history=conversation_history, df=df, browser_service=browser_service)
        logger.debug(f"Chat processing result: {result}")

        response = make_response(result)
        origin = request.headers.get('Origin')
        logger.debug(f"Request origin: {origin}")

        if origin in ALLOWED_ORIGINS:
            logger.info(f"Origin {origin} is allowed. Adding CORS headers.")
            response.headers.update({
                'Access-Control-Allow-Origin': origin,
                'Access-Control-Allow-Credentials': 'true'
            })
        else:
            logger.warning(f"Origin {origin} is not in allowed origins.")

        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)

        error_response = jsonify({
            "error": str(e),
            "message": "Failed to process chat request"
        })
        error_response.status_code = 500

        logger.debug(f"Error response: {error_response.get_json()}")
        return error_response


# Socket event handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connect', {'status': 'connected', 'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect(data=None):
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('browse')
def handle_browse(data):
    url = data.get('url')
    try:
        result = browser_service.navigate_to(url)
        socketio.emit('browse_result', result)
    except Exception as e:
        logger.error(f"Browse error: {str(e)}")
        socketio.emit('error', {'message': f'Failed to browse: {str(e)}'})


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
    df = embedding_tool.initialize_directory_embeddings('xyz/modules/llm/embedding_tools/embeddings/source_documents',
                                       'source_documents', update=True)
    return jsonify({"status": "success", "message": "Embeddings updated"})


# Register the LLM blueprint
init_llm_bp(app)

if __name__ == '__main__':
    ensure_directory_exists(EMBEDDINGS_DIR)

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
