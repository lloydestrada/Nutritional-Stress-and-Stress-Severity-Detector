from flask import Flask, request, jsonify, send_file, render_template
from rembg import remove
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load model and class labels once at the start
MODEL_PATH = 'model/nutrient_cnn_model.h5'
CLASS_LABELS = ['healthy', 'nitrogen-N', 'phosphorus-P', 'potasium-K']
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        input_image = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Remove background
    output_image = remove(input_image)

    # Convert PIL image to OpenCV format
    image_np = np.array(output_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Create a binary mask to isolate the leaf
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, leaf_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Erode the leaf mask to remove edges
    kernel = np.ones((5, 5), np.uint8)
    eroded_leaf_mask = cv2.erode(leaf_mask, kernel, iterations=3)

    # Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Convert to HSV and YCrCb color spaces
    hsv = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2YCrCb)

    # Define HSV range for healthy green leaves
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Extract Cr (YCrCb) and A (LAB) channels
    cr_channel = ycrcb[:, :, 1]
    a_channel = lab[:, :, 1]

    # Detect diseased areas
    non_green_mask = cv2.bitwise_not(healthy_mask)
    non_green_mask = cv2.medianBlur(non_green_mask, 5)

    _, disease_mask_cr = cv2.threshold(cr_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, disease_mask_a = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    disease_mask = cv2.bitwise_or(disease_mask_cr, disease_mask_a)
    disease_mask = cv2.bitwise_and(disease_mask, non_green_mask)
    disease_mask = cv2.bitwise_and(disease_mask, eroded_leaf_mask)

    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Highlight diseased areas
    image_highlighted = image_np.copy()
    image_highlighted[disease_mask == 255] = [255, 0, 0]

    # Convert image back to PIL for return
    image_highlighted_pil = Image.fromarray(image_highlighted)
    img_byte_arr = io.BytesIO()
    image_highlighted_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')


@app.route('/predict_deficiency', methods=['POST'])
def predict_deficiency():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = CLASS_LABELS[predicted_index]

        return jsonify({'class': predicted_label})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
