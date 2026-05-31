

import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ── App create karo ──────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Model load karo (sirf ek baar) ───────────
print("Model load ho raha hai...")
model = load_model('final_model.keras')
print("✅ Model ready!")

# ── Image predict karne ka function ──────────
def predict_image(img_path):
    # Image open karo aur resize karo
    img = image.load_img(img_path, target_size=(150, 150))

    # Numbers mein convert karo
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Model se prediction lo
    prediction = model.predict(img_array)[0][0]

    # 0.5 se upar = Dog, neeche = Cat
    if prediction > 0.5:
        label = "Dog 🐶"
        confidence = round(float(prediction) * 100, 1)
    else:
        label = "Cat 🐱"
        confidence = round((1 - float(prediction)) * 100, 1)

    return label, confidence

# ── Routes ───────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # File check karo
    if 'file' not in request.files:
        return jsonify({'error': 'Koi file nahi mili!'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'File select nahi ki!'})

    # File save karo
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict karo
    label, confidence = predict_image(filepath)

    # Result wapas bhejo
    return jsonify({
        'label': label,
        'confidence': confidence
    })

# ── Start Server
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)

