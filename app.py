from flask import Flask
import os
import logging
from routes import main
from auth import auth

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # change this securely

# Config
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Routes
app.register_blueprint(auth)
app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True)