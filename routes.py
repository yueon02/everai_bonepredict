from flask import Blueprint, render_template, redirect, url_for, session, request, jsonify, current_app
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging
import random

main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# Optional: Set random seed for consistent predictions
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load models
try:
    gender_model = tf.keras.models.load_model('model.py/best_model.h5')
    xray_classifier = tf.keras.models.load_model('model.py/xray_classifier.h5')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

def allowed_file(filename, app):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    img = tf.keras.utils.load_img(
        img_path,
        color_mode='rgb',
        target_size=target_size
    )
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def is_xray_image(img_path):
    """Check if the uploaded image is an X-ray using the classifier"""
    try:
        processed_img = preprocess_image(img_path, target_size=(128, 128))
        prediction = xray_classifier.predict(processed_img)
        is_xray = prediction[0][0] > 0.7
        confidence = float(prediction[0][0]) if is_xray else float(1 - prediction[0][0])
        return is_xray, confidence
    except Exception as e:
        logger.error(f"X-ray classification failed: {str(e)}", exc_info=True)
        return False, 0.0

@main.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('auth.login'))
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    if 'xray' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['xray']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not (file and allowed_file(file.filename, current_app)):
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

    # Add randomness to filename to avoid overwrite
    unique_id = str(np.random.randint(100000, 999999))
    filename = secure_filename(f"{unique_id}_{file.filename}")
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)

        # 1. Check if it's a valid X-ray
        is_xray, xray_confidence = is_xray_image(filepath)
        if not is_xray:
            raise ValueError(
                f"The uploaded image doesn't appear to be an X-ray (confidence: {xray_confidence:.1%}). "
                "Please upload a valid skull X-ray image."
            )

        # 2. Preprocess for gender model
        processed_img = preprocess_image(filepath, target_size=(224, 224))

        # 3. Predict gender
        pred = gender_model.predict(processed_img)

        if pred.size == 0:
            raise ValueError("Gender model returned an empty prediction.")

        pred_prob = float(pred[0][0])
        gender = 'female' if pred_prob > 0.5 else 'male'
        confidence = round(pred_prob * 100, 1) if gender == 'female' else round((1 - pred_prob) * 100, 1)

        return jsonify({
            "success": True,
            "gender": gender,
            "confidence": confidence,
            "filename": filename,
            "xray_verification": {
                "is_xray": True,
                "confidence": round(xray_confidence * 100, 1)
            }
        })

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            "success": False,
            "error": str(ve),
            "hint": "Please upload a valid skull X-ray image (JPEG/PNG)"
        }), 400

    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "hint": "Our system encountered an error. Please try again."
        }), 500

    finally:
        # Cleanup file
        if os.path.exists(filepath):
            os.remove(filepath)

