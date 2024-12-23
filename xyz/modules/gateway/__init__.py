from flask import Flask
from config import Config
import xyz.modules.gateway.gateway_blueprint


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)

    return app
