# app.py (Decoy Blog) hosted on heroku at https://isadora-v2-74e5a1b97f07.herokuapp.com
import eventlet
from datetime import datetime, timedelta
from collections import defaultdict
import os
import re
import sys
import tempfile
import ssl
import urllib3
from flask import Flask, request, make_response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask import Flask, request, jsonify, session, render_template, abort, redirect
import psycopg2
from threading import Thread
import socketio

from werkzeug.serving import WSGIRequestHandler
import pandas as pd
import numpy as np
import geoip2.database

sys.setrecursionlimit(3000)
import config
from config import SK, OAI, AIS, logger, gateway
from xyz.modules.llm.embedding_tools.fingerprint_embeddings import fingerprint_bot
from xyz.modules.llm import embedding_tool



def create_ssl_context():
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    except Exception as e:
        logger.error(f"Failed to create SSL context: {e}")
        return None


ssl_context = create_ssl_context()
conversation_history = {}
# In-memory rate-limiting storage (can be replaced with Redis or a database)
RATE_LIMIT = defaultdict(list)
app = Flask(__name__)
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)

WSGIRequestHandler.timeout = 600

# Explicitly define allowed origins
ALLOWED_ORIGINS = [
    'https://isadora-f5fbebf38bc6.herokuapp.com',
    'https://isadora-v2-74e5a1b97f07.herokuapp.com',
    'https://34.16.120.105',
    'https://isadora.ai',
    'https://io.isadora.ai',
    'https://chat.isadora.ai',
    'https://73.18.165.209',
    'https://64.44.118.215',
]


CORS(app,
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With",
                   "Accept", "Origin", "Access-Control-Request-Method",
                   "Access-Control-Request-Headers"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])


# Configure SocketIO with explicit CORS settings
socketio_app = SocketIO(
    app,
    cors_allowed_origins=ALLOWED_ORIGINS,
    async_mode='eventlet',
    ping_timeout=60000,
    ping_interval=25000,
    always_connect=True,
    path='/socket.io',
    transport=['websocket'],  # Ensure 'websocket' is included
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


@app.before_request
def log_request():
    """Log incoming requests."""
    logger.info(
        f"Incoming request: {request.method} {request.path} | "
        f"Headers: {dict(request.headers)} | "
        f"Body: {request.get_json(silent=True)}"
    )

# Middleware to block suspicious requests and fingerprint bots
@app.before_request
def block_suspicious_requests():
    origin = request.headers.get("X-Real-Ip") or request.headers.get("Origin")
    if origin:
        origin = origin.split(",")[0].strip()  # Normalize origin
        if not origin.startswith("http://") and not origin.startswith("https://"):
            origin = f"https://{origin}"  # Add schema if missing

    heroku_secret = request.headers.get("X-Heroku-Auth")
    user_agent = request.headers.get("User-Agent", "").lower()
    ip_address = request.headers.get("X-Real-Ip", request.remote_addr)
    ip_address = ip_address.split(",")[0].strip()  # Normalize ip



    logger.info(f"X-Real-IP: {ip_address}")
    timestamp = datetime.utcnow()

    # 1. Check Trusted Heroku App Header
    if heroku_secret == SK:  # Replace with your Heroku secret
        app.logger.info(f"Trusted request from Heroku app.")
        return  # Allow the request

    # 2. Block Suspicious Patterns in Request Paths
    for pattern in AIS.SUSPICIOUS_PATTERNS:
        if re.search(pattern, request.path, re.IGNORECASE):
            app.logger.info(f"Blocked suspicious request to: {request.path}")
            fingerprint_bot(request, ip_address, user_agent)  # Fingerprint the suspicious bot
            abort(403)  # Return a 403 Forbidden response

    # 3. Block Suspicious User-Agent Strings
    for pattern in AIS.BLOCKED_USER_AGENTS:
        if re.search(pattern, user_agent, re.IGNORECASE):
            app.logger.info(f"Blocked suspicious User-Agent: {user_agent}")
            fingerprint_bot(request, ip_address, user_agent)  # Fingerprint the suspicious bot
            abort(403)

    # 4. Rate Limiting
    RATE_LIMIT[ip_address].append(timestamp)
    # Keep only requests in the last minute
    RATE_LIMIT[ip_address] = [t for t in RATE_LIMIT[ip_address] if t > timestamp - timedelta(minutes=1)]
    if len(RATE_LIMIT[ip_address]) > AIS.MAX_REQUESTS_PER_MINUTE:  # Set your rate limit
        app.logger.info(f"Rate limit exceeded for IP: {ip_address}")
        fingerprint_bot(request, ip_address, user_agent)  # Fingerprint the suspicious bot
        abort(429)  # Too Many Requests

    # 5. Geo-Location Blocking
    try:
        with geoip2.database.Reader('/path/to/GeoLite2-City.mmdb') as reader:
            geo_data = reader.city(ip_address)
            country = geo_data.country.iso_code
            if country in AIS.BLOCKED_COUNTRIES:
                app.logger.info(f"Blocked request from blocked country: {country}")
                fingerprint_bot(request, ip_address, user_agent)  # Fingerprint the suspicious bot
                abort(403)
    except Exception as e:
        app.logger.error(f"Geo-location lookup failed: {e}")

    # 6. Honeypot Detection
    if request.path in AIS.HONEYPOT_ENDPOINTS:
        app.logger.warning(f"Honeypot triggered by IP: {ip_address}")
        fingerprint_bot(request, ip_address, user_agent)  # Fingerprint the suspicious bot
        abort(403)

    # 7. Header Inconsistencies
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    x_real_ip = request.headers.get("X-Real-Ip")
    if x_forwarded_for and x_real_ip and x_real_ip not in x_forwarded_for:
        app.logger.info(f"Blocked request with inconsistent headers from IP: {ip_address}")
        fingerprint_bot(request, ip_address, user_agent)  # Fingerprint the suspicious bot
        abort(403)

    # 8. TLS/SSL Analysis (Optional)
    # You can analyze TLS/SSL protocol and cipher suite here if needed
    # Example: Check for outdated TLS versions or weak cipher suites

    # 9. Block All Origins
    if gateway == 'closed':
        if origin not in ALLOWED_ORIGINS:
            app.logger.info(f"Blocked request from disallowed origin: {origin}")
            abort(403)

    # 10. Log All Requests for Monitoring
    app.logger.info(f"Request Path: {request.path}, User-Agent: {user_agent}, IP: {ip_address}")


@app.before_request
def handle_options_preflight():
    if request.method == 'OPTIONS':
        response = app.make_response('')
        response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response, 204


@app.route('/')
def index():
    logger.info("Serving index page.")
    return render_template('index.html')


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


# Socket event handlers
@socketio_app.on('connect')
def handle_connect():
    #logger.info(f"Client connected: {request.sid}")
    emit('connect', {'status': 'connected', 'sid': request.sid})


@socketio_app.on('disconnect')
def handle_disconnect(data=None):
    #logger.info(f"Client disconnected: {request.sid}")
    x = 100


@socketio_app.on('typing')
def handle_typing(data):
    try:
        emit('typing', data, broadcast=True, include_self=False)
    except Exception as e:
        logger.error(f"Error in typing event: {e}")
        emit('error', {'message': 'Failed to broadcast typing status'})


@socketio_app.on('stop_typing')
def handle_stop_typing(data):
    try:
        emit('stop_typing', data, broadcast=True, include_self=False)
    except Exception as e:
        logger.error(f"Error in stop_typing event: {e}")
        emit('error', {'message': 'Failed to broadcast stop typing status'})


@socketio_app.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {e}")
    return {"error": str(e)}


@socketio_app.on_error_default
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


if __name__ == '__main__':

    if config.Config.production:
        socketio_app.run(
            app,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),  # Use Heroku's assigned port
            debug=False,
            use_reloader=False,
            cors_allowed_origins=ALLOWED_ORIGINS
        )
    else:
        socketio_app.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            ssl_context=ssl_context,
            cors_allowed_origins=ALLOWED_ORIGINS
        )

