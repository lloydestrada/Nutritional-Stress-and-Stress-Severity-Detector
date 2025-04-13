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

# Load nutrient deficiency model
NUTRIENT_MODEL_PATH = 'model/nutrient_cnn_model.h5'
NUTRIENT_CLASS_LABELS = ['healthy', 'nitrogen-N', 'phosphorus-P', 'potassium-K']
nutrient_model = tf.keras.models.load_model(NUTRIENT_MODEL_PATH)

# Load severity model
SEVERITY_MODEL_PATH = 'model/Severity_Main1.h5'
SEVERITY_CLASS_LABELS = ['Healthy', 'High', 'Low', 'Medium']
severity_model = tf.keras.models.load_model(SEVERITY_MODEL_PATH)

recommendations = {
    "healthy_healthy": "Your coffee plant is healthy. No action needed.",
    "nitrogen-n_low": "Apply a nitrogen-rich fertilizer (e.g., urea or ammonium sulfate) and incorporate organic compost to improve soil fertility.",
    "nitrogen-n_medium": "Supplement nitrogen with organic manure or foliar sprays. Monitor leaf color over time.",
    "nitrogen-n_high": "Apply quick-release nitrogen sources and prune affected leaves to prevent spread.",
    "phosphorus-p_low": "Use phosphorus-based fertilizers like superphosphate. Ensure soil pH is within 6-7 for better absorption.",
    "phosphorus-p_medium": "Use rock phosphate or composted manure to gradually boost phosphorus levels.",
    "phosphorus-p_high": "Apply phosphoric acid foliar sprays and remove severely affected leaves.",
    "potassium-k_low": "Apply potassium sulfate or wood ash. Water consistently to improve nutrient uptake.",
    "potassium-k_medium": "Use balanced NPK fertilizers with higher K ratio. Avoid water stress.",
    "potassium-k_high": "Incorporate muriate of potash (KCl) and use mulch to retain soil moisture.",
    "healthy_high": "No nutrient issues, but signs of high stress. Consider improving shading, irrigation, and pest control.",
    "healthy_medium": "Moderate stress detected. Monitor regularly for emerging symptoms.",
    "healthy_low": "Mild stress detected. Maintain consistent care and nutrients."
}

@app.route('/')
def index():
    return render_template('index.html')

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

        predictions = nutrient_model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = NUTRIENT_CLASS_LABELS[predicted_index]

        return jsonify({'class': predicted_label})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_severity', methods=['POST'])
def predict_severity():
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

        predictions = severity_model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = SEVERITY_CLASS_LABELS[predicted_index]

        return jsonify({'severity_class': predicted_label})

    except Exception as e:
        return jsonify({'error': f'Severity prediction failed: {str(e)}'}), 500

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    nutrient = data.get("nutrient")
    severity = data.get("severity")
    if not nutrient or not severity:
        return jsonify({"error": "Missing nutrient or severity value"}), 400

    key = f"{nutrient.lower()}_{severity.lower()}"
    recommendation = recommendations.get(key, "No specific recommendation found.")
    return jsonify({"recommendation": recommendation})

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

    kernel = np.ones((5, 5), np.uint8)
    eroded_leaf_mask = cv2.erode(leaf_mask, kernel, iterations=3)

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    hsv = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2YCrCb)

    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)

    cr_channel = ycrcb[:, :, 1]
    a_channel = lab[:, :, 1]

    non_green_mask = cv2.bitwise_not(healthy_mask)
    non_green_mask = cv2.medianBlur(non_green_mask, 5)

    _, disease_mask_cr = cv2.threshold(cr_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, disease_mask_a = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    disease_mask = cv2.bitwise_or(disease_mask_cr, disease_mask_a)
    disease_mask = cv2.bitwise_and(disease_mask, non_green_mask)
    disease_mask = cv2.bitwise_and(disease_mask, eroded_leaf_mask)

    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    image_highlighted = image_np.copy()
    image_highlighted[disease_mask == 255] = [255, 0, 0]

    image_highlighted_pil = Image.fromarray(image_highlighted)
    img_byte_arr = io.BytesIO()
    image_highlighted_pil.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)