import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = load_model('C:\\Users\\kiran\\Desktop\Projects\\face_recognization\\face_recognition_model.keras')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "No file uploaded"

    # Read the uploaded file
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image (resize, convert to array, normalize)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    # Make prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))

    # Assuming the model returns probabilities for each class
    class_names = ['person1', 'person2']  
    predicted_class = class_names[np.argmax(prediction)]

    return f"Predicted Class: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
