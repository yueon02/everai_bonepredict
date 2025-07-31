from flask import Flask
import os
import logging
from routes import main
from auth import auth

def create_app():
    app = Flask(__name__)
    app.secret_key = 'supersecretkey'  # Change to secure secret key

    # Config
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register blueprints
    app.register_blueprint(auth)
    app.register_blueprint(main)

    return app

# Local run (optional, not used by Render)
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=10000)
