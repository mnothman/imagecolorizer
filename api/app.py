import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow as tf 
app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', '../models/colorization_model_epoch_21.keras')
# might need tf.keras.models.load_model() to load the model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def home():
    return app.send_static_file("app.html")

@app.route("/colorize", methods=["POST"])
def colorize():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    
    # First decode as grayscale
    gray_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        return jsonify({"error": "Invalid image"}), 400
    
    # Resize and normalize
    gray_img_resized = cv2.resize(gray_img, (256, 256))
    gray_img_normalized = gray_img_resized.astype(np.float32) / 255.0
    gray_img_normalized = np.expand_dims(gray_img_normalized, axis=(0, -1))  # shape (1, 256, 256, 1)

    # Predict color
    pred = model.predict(gray_img_normalized)[0]  # shape (256, 256, 3)
    pred = np.clip(pred * 255.0, 0, 255).astype('uint8')

    # Encode resulting image from above
    _, buffer = cv2.imencode('.png', pred)
    encoded_string = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"colorized_image": encoded_string})

@app.route("/grayscale", methods=["POST"])
def grayscale():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)

    color_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if color_img is None:
        return jsonify({"error": "Invalid image"}), 400

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Encode resulting image
    _, buffer = cv2.imencode('.png', gray_img)
    encoded_string = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"grayscale_image": encoded_string})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
