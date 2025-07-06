import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# Set path
DATASET_PATH = 'data/genres_original'

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# def extract_features(file_path):
#     y, sr = librosa.load(file_path, mono=True, duration=30)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#     return mfcc

def extract_features(file_path, max_pad_len=1300):
    y, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Padding or truncating to fixed length
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc


features = []
labels = []

for label, genre in enumerate(genres):
    genre_path = os.path.join(DATASET_PATH, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        mfcc = extract_features(file_path)
        mfcc = np.expand_dims(mfcc, axis=-1)
        features.append(mfcc)
        labels.append(label)

X = np.array(features)
y = to_categorical(np.array(labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('model/music_genre_classifier_model.h5')
