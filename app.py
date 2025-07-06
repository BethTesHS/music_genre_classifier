from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
from keras.models import load_model

app = Flask(__name__)
CORS(app)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
model = load_model('model/music_genre_classifier_model.h5')

# def predict_genre(file_path):
#     y, sr = librosa.load(file_path, mono=True, duration=30)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     mfcc = np.expand_dims(mfcc, axis=-1)
#     mfcc = np.expand_dims(mfcc, axis=0)
#     prediction = model.predict(mfcc)[0]
#     top5_indices = prediction.argsort()[-5:][::-1]
#     top5_genres = [(genres[i], float(prediction[i])) for i in top5_indices]
#     return top5_genres

def predict_genre(file_path, max_pad_len=1300):
    y, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Ensure consistent shape:
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    mfcc = np.expand_dims(mfcc, axis=-1)   # (40, 1300, 1)
    mfcc = np.expand_dims(mfcc, axis=0)    # (1, 40, 1300, 1)
    prediction = model.predict(mfcc)[0]
    top5_indices = prediction.argsort()[-5:][::-1]
    top5_genres = [(genres[i], float(prediction[i])) for i in top5_indices]
    return top5_genres


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = 'temp/temp_audio.wav'
    file.save(file_path)

    top5 = predict_genre(file_path)
    return jsonify({'predictions': top5})

if __name__ == '__main__':
    app.run(debug=True)
