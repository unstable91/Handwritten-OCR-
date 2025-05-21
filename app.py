import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime

# Initialize flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#Load model
model = tf.keras.models.load_model(r'C:\Users\rupes\PycharmProjects\EMNIST\.venv\emnist_ocr_model.keras')
print("Model loaded successfully!")

#Load character mapping
mapping = {}
with open(r"C:\Users\rupes\Downloads\gzip\gzip\emnist-balanced\emnist-balanced-mapping.txt", 'r') as f:
    for line in f:
        label, ascii_code = map(int, line.split())
        mapping[label] = chr(ascii_code)

# Recent predictions 
recent_predictions = []  # dict: { image_path, label, timestamp }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  #grayscale
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocessing and predict
        preprocessed_image = preprocess_image(file_path)
        prediction = model.predict(preprocessed_image)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_char = mapping.get(predicted_label, '?')

        label_str = f"{predicted_char} (Class {predicted_label})"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        #Store recent prediction
        recent_predictions.insert(0, {
            'image_path': f'uploads/{filename}',
            'label': label_str,
            'timestamp': timestamp
        })
        recent_predictions[:] = recent_predictions[:10]  #max 10

        return render_template('result.html',
                               label=label_str,
                               image_path=f'uploads/{filename}',
                               recent_predictions=recent_predictions)

    #Return upload page and pass recent predictions
    return render_template('index.html', recent_predictions=recent_predictions)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
