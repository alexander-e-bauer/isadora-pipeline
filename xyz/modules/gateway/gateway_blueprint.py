from flask import Blueprint, render_template, request, jsonify

gw = Blueprint('gateway', __name__, template_folder='templates')


def init_app(app):
    app.register_blueprint(gw)


@gw.route('/')
def gateway():
    return render_template('gateway.html')


@gw.route('/index')
def lens():
    return render_template('index.html')


