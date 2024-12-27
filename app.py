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

eventlet.monkey_patch(thread=False)  # Add thread=False to help with recursion issues
sys.setrecursionlimit(3000)

from xyz.modules.gateway import gateway_blueprint
from xyz.modules.llm import llm_blueprint, embedding_tool
from xyz.modules.database import database, models
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

# Configure CORS - Moved to top and expanded configuration
CORS(app, resources={
    r"/*": {  # Changed from r"/api/*" to r"/*"
        "origins": [
            "http://localhost:3000",
            "http://localhost:5000",
            "http://localhost:6379",
            "https://isadora-v2-74e5a1b97f07.herokuapp.com",
            "https://isadora-f5fbebf38bc6.herokuapp.com"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configure SocketIO with expanded CORS settings
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
    always_connect=True
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
gw = gateway_blueprint.init_app(app)
oai = llm_blueprint.init_app(app)

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

# Routes with updated CORS handling
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

    data = request.json
    result = embedding_tool.jsonify_chat(data, conversation_history)
    response = make_response(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
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
            use_reloader=False
        )
    else:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            ssl_context=ssl_context,
            allow_unsafe_werkzeug=True
        )
