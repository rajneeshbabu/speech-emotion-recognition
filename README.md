# Speech Emotion Recognition (SER)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Detects 8 emotions from speech audio using 6 ML models trained on the RAVDESS dataset.**

[Live Demo](https://speech-emotion-recognition-rajneesh.streamlit.app) • [Dataset](https://zenodo.org/record/1188976)

</div>

---

## Results

| Model         | Accuracy | F1 Weighted | ROC-AUC |
|--------------|:--------:|:-----------:|:-------:|
| **CNN**      | **89.58%** | **0.8964** | 0.9894  |
| **SVM**      | **89.58%** | 0.8949     | **0.9926** |
| Bi-LSTM      | 85.76%   | 0.8579      | 0.9761  |
| XGBoost      | 85.76%   | 0.8573      | 0.9905  |
| Random Forest| 84.90%   | 0.8480      | 0.9914  |
| CNN-LSTM     | 81.42%   | 0.8111      | 0.9624  |

CNN and SVM tied at 89.58% accuracy. CNN wins on F1-weighted, SVM wins on ROC-AUC.
Results from `artifacts/model_comparison.json` — RAVDESS test set (288 samples, 20% holdout).

---

## Emotions Detected

neutral · calm · happy · sad · angry · fearful · disgust · surprised

---

## Architecture

```
Raw Audio (.wav)
      |
      v
Preprocessing
  - Silence trimming (librosa)
  - Spectral noise gating
  - Resampling to 22050 Hz
      |
      |-------------|
      v             v
Flat Features    2D Features
  MFCC x3         MFCC (40 x 128)
  Mel Spectrogram  Mel  (128 x 128)
  Chroma
  ZCR, RMS
  Spectral Contrast
  Tonnetz
  (~315 dims)
      |             |
      v             v
Classical ML    Deep Learning
  PCA (95%)     CNN
  RandomForest  Bi-LSTM
  SVM           CNN-LSTM
  XGBoost
```

---

## Project Structure

```
Speech Emotion Recognition/
├── notebooks/
│   └── SER_Training.ipynb      <- run this to train all 6 models
├── models/                     <- saved model files (committed to git)
│   ├── cnn.keras
│   ├── lstm.keras
│   ├── cnn_lstm.keras
│   ├── random_forest.pkl
│   ├── svm.pkl
│   └── xgboost.pkl
├── artifacts/
│   ├── processed/
│   │   └── label_map.pkl       <- emotion label mapping
│   └── plots/                  <- confusion matrices, ROC curves, SHAP
├── app.py                      <- Streamlit web app
├── requirements.txt
├── packages.txt                <- system deps for Streamlit Cloud
└── .gitignore
```

---

## Run Locally

```bash
# Clone and set up virtual environment
git clone https://github.com/rajneeshbabu/speech-emotion-recognition.git
cd speech-emotion-recognition

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Launch the app (models already included in the repo)
python -m streamlit run app.py
```

Open http://localhost:8501

---

## Train from Scratch

If you want to retrain the models yourself:

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Open the training notebook
jupyter notebook notebooks/SER_Training.ipynb
```

Run all cells. The notebook downloads RAVDESS (~3.5 GB), extracts features, trains all 6 models, and saves them to `models/`.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy**

`packages.txt` handles the system dependencies (`libsndfile1`, `ffmpeg`) automatically.

---

## Features

| Feature | Dimensions | Description |
|---------|:----------:|-------------|
| MFCC mean + std | 80 | Mel-frequency cepstral coefficients |
| MFCC delta | 40 | First-order derivatives |
| MFCC delta2 | 40 | Second-order derivatives |
| Mel Spectrogram | 128 | Log-power mel spectrogram |
| Chroma | 12 | Pitch class profile |
| ZCR | 1 | Zero-crossing rate |
| RMS Energy | 1 | Loudness |
| Spectral Contrast | 7 | Spectral valley-to-peak ratio |
| Tonnetz | 6 | Tonal centroid features |
| **Total** | **~315** | |

---

Built by [Rajneesh](https://github.com/rajneeshbabu) · MIT License
