from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import librosa
import os

app = Flask(__name__)
CORS(app)

# Load the trained raw waveform model
model = tf.keras.models.load_model('models/music_genre_raw_model.h5')

# Genre labels
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# Constants
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Load audio
        y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)

        # Pad or truncate
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
        else:
            y = y[:SAMPLES_PER_TRACK]

        # Reshape for prediction
        y = np.expand_dims(y, axis=-1)  # (661500, 1)
        y = np.expand_dims(y, axis=0)   # (1, 661500, 1)

        # Predict
        prediction = model.predict(y)[0]  # Shape: (10,)

        # Get top 5 predictions
        top5_indices = prediction.argsort()[-5:][::-1]
        top5 = [(genres[i], float(prediction[i])) for i in top5_indices]

        return jsonify({'predictions': top5})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Error during prediction'})

if __name__ == '__main__':
    app.run(debug=True)
