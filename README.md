# Music Genre Classification Project

This project detects and classifies **music genres** using **machine learning** on the **GTZAN dataset**. A user uploads an audio file through a web interface, and the system predicts the **top 5 possible genres**.

---

## Features

✅ Upload music files via a web interface.  
✅ Trained CNN model on GTZAN dataset.  
✅ Predicts top 5 genres for uploaded music.  

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/BethTesHS/music_genre_classifier
cd music_genre_classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install required libraries

```bash
pip install tensorflow keras librosa numpy pandas scikit-learn flask flask-cors
```

### 4. Preparing the Dataset
#### Download the GTZAN dataset from:
- [GTZAN Music Genre Classification Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

#### Place the extracted folder inside:
```bash
music_genre_classifier/data/
```

### Running the Training Script
```bash
python train.py
```

### Running the Flask App
#### To start your web interface for uploading and predicting:
```bash
python app.py
```

#### After starting, open your browser and navigate to:
```bash
http://127.0.0.1:5000/
```
