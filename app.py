from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import cv2
import os
from PIL import Image
import io
import base64
import uuid
import datetime
import time
from concurrent.futures import ThreadPoolExecutor 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
IMG_SIZE = (224, 224)
EPSILON = 0.04  # Perturbation strength for FGSM attack

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = preprocess_input(img)
    return img

def generate_fgsm_attack(img, epsilon=EPSILON):
    img_tensor = tf.convert_to_tensor(img)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor[np.newaxis, ...])
        loss = tf.keras.losses.MSE(tf.ones_like(prediction), prediction)
    
    gradient = tape.gradient(loss, img_tensor)
    perturbation = epsilon * tf.sign(gradient)
    adversarial_img = img_tensor + perturbation
    adversarial_img = tf.clip_by_value(adversarial_img, -1, 1)
    return adversarial_img.numpy()

def detect_adversarial(img):
    gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    threshold = 100
    return variance > threshold

def defend_model(img, method='gaussian'):
    """Apply defense mechanism to image"""
    # Convert to 0-255 range for OpenCV processing
    img_uint8 = ((img + 1) * 127.5).astype('uint8')
    
    if method == 'gaussian':
        defended = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    else:  # Add other methods if needed
        defended = img_uint8.copy()
    
    # Convert back to normalized range
    return (defended.astype('float32') / 127.5) - 1.0

def decode_prediction_string(predictions):
    decoded = decode_predictions(predictions)[0]
    return ', '.join([f"{p[1]} ({p[2]:.2f})" for p in decoded])

def image_to_base64(img):
    """Convert numpy array to base64 encoded image"""
    if isinstance(img, Image.Image):
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
    
    if img.min() < 0:  # If normalized [-1,1]
        img = (img + 1) * 127.5
    img = np.clip(img, 0, 255).astype('uint8')
    img_pil = Image.fromarray(img)
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def format_predictions(preds, top=5):
    """Format model predictions for display"""
    decoded = decode_predictions(preds, top=top)[0]
    return {p[1]: float(p[2]) for p in decoded}

def prediction_str(pred_dict):
    """Convert prediction dictionary to string"""
    return ", ".join([f"{k} ({v:.2f})" for k, v in pred_dict.items()])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return render_template('generate.html')
    
    # Handle POST request for generating attacks
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        original_img = preprocess_image(img)
        adversarial_img = generate_fgsm_attack(original_img, float(request.form.get('epsilon', EPSILON)))
        
        # Generate difference visualization
        difference = np.abs(adversarial_img - original_img) * 10  # Amplify differences
        difference = (difference * 255).astype(np.uint8)
        difference_img = Image.fromarray(difference)
        
        # Make predictions
        original_pred = model.predict(original_img[np.newaxis, ...])
        adversarial_pred = model.predict(adversarial_img[np.newaxis, ...])

        # Save images to disk
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Create filenames
        original_filename = f"original_{timestamp}_{unique_id}.jpg"
        adversarial_filename = f"adversarial_{timestamp}_{unique_id}.jpg"
        difference_filename = f"difference_{timestamp}_{unique_id}.jpg"
        
        # Save images
        original_img_pil = Image.fromarray((original_img * 127.5 + 127.5).astype(np.uint8))
        adversarial_img_pil = Image.fromarray((adversarial_img * 127.5 + 127.5).astype(np.uint8))
        difference_img_pil = Image.fromarray(difference)
        
        original_img_pil.save(os.path.join(app.config['UPLOAD_FOLDER'], original_filename))
        adversarial_img_pil.save(os.path.join(app.config['UPLOAD_FOLDER'], adversarial_filename))
        difference_img_pil.save(os.path.join(app.config['UPLOAD_FOLDER'], difference_filename))

        return jsonify({
            'original_prediction': decode_prediction_string(original_pred),
            'adversarial_prediction': decode_prediction_string(adversarial_pred),
            'original_image': image_to_base64(original_img),
            'adversarial_image': image_to_base64(adversarial_img),
            'difference_image': image_to_base64(difference)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html')
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_array = np.array(img)
        
        # Convert image to proper format for detection
        img_uint8 = (img_array * 255).astype(np.uint8) if img_array.max() <= 1 else img_array
        
        # Detection logic
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        threshold = 100
        is_adversarial = bool(variance > threshold)
        
        # Generate visualizations
        freq = np.fft.fft2(gray)
        freq_shift = np.fft.fftshift(freq)
        magnitude = 20 * np.log(np.abs(freq_shift) + 1e-10)  # Added small value to avoid log(0)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)
        
        return jsonify({
            'is_adversarial': is_adversarial,  # Now properly serialized
            'adversarial_score': float(variance / threshold),  # Convert to float
            'laplacian_variance': float(variance),
            'feature_inconsistency': float(0.75 if is_adversarial else 0.25),
            'prediction_confidence': float(0.9),
            'frequency_analysis': image_to_base64(magnitude),
            'gradient_visualization': image_to_base64(gradient)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/defend', methods=['GET', 'POST'])
def defend():
    print("Defend route hit!")  # Debugging line
    if request.method == 'GET':
        return render_template('defend.html')
    
    if 'image' not in request.files:
        print(f"File received: {file.filename}") # debugging line
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Load image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_array = np.array(img)
        
        # Assume input is already perturbed, apply defense
        defended_img = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # Preprocess for model
        perturbed_preprocessed = preprocess_input(img_array.astype('float32'))
        defended_preprocessed = preprocess_input(defended_img.astype('float32'))
        
        # Get predictions
        perturbed_pred = model.predict(perturbed_preprocessed[np.newaxis, ...])
        defended_pred = model.predict(defended_preprocessed[np.newaxis, ...])
        
        # Format predictions
        def format_pred(pred):
            decoded = decode_predictions(pred)[0]
            return {p[1]: float(p[2]) for p in decoded[:3]}  # Top 3 predictions
        
        # Convert images to base64
        def img_to_base64(img_array):
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        
        return jsonify({
            'perturbed_image': img_to_base64(img_array),
            'defended_image': img_to_base64(defended_img),
            'perturbed_pred': format_pred(perturbed_pred),
            'defended_pred': format_pred(defended_pred),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
