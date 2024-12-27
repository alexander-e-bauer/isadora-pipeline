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

# Allowed origins
ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'https://isadora-v2-74e5a1b97f07.herokuapp.com',
    'https://isadora-f5fbebf38bc6.herokuapp.com'
]

# CORS Configuration matching React app requirements
CORS(app, resources={
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
            "Authorization",
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Credentials"
        ],
        "supports_credentials": True,
        "send_wildcard": False,
        "max_age": 86400
    }
})

# SocketIO Configuration matching React app socket config
socketio = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode='eventlet',
    ping_timeout=60,  # Matching frontend timeout
    ping_interval=15,
    always_connect=True,
    path='/socket.io',
    transport=['websocket', 'polling'],
    cookie=False,
    cors_credentials=True
)

# Flask configuration
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")

# Database configuration
username_db = os.getenv('POSTGRES_USER')
password_db = os.getenv('POSTGRES_PASSWORD')
database_name = os.getenv('POSTGRES_DB')
postgres_uri = f'postgresql://{username_db}:{password_db}@localhost/{database_name}'
app.config['SQLALCHEMY_DATABASE_URI'] = postgres_uri

app.config['SECURITY_PASSWORD_SALT'] = os.environ.get("SECURITY_PASSWORD_SALT")

# Create database connection object
db = database.init_app(app)

# Define models
User, Role, roles_users = models.define_models(db)

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
#gw = gateway_blueprint.init_app(app)
oai = llm_blueprint.init_app(app)

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        response.headers['Access-Control-Max-Age'] = '86400'
    return response

# Updated chat route with proper CORS and error handling
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = make_response()
        origin = request.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
            response.headers['Access-Control-Max-Age'] = '86400'
        return response

    try:
        data = request.json
        result = embedding_tool.jsonify_chat(data, conversation_history)
        response = make_response(result)
        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        error_response = jsonify({
            "error": str(e),
            "message": "Failed to process chat request"
        })
        error_response.status_code = 500
        return error_response

# Socket.IO event handlers with error handling
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connect', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
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