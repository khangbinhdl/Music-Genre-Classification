# 🎵 Music Genre Classification Inference Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MuqPbDQ7LmrrYWrljAdMmMkVVzNj2v6z?usp=drive_link)

Train models from GTZAN `features_3_sec.csv` to predict music genres from 3-second audio features.

This repository is a production-style refactor of the original Colab notebook above, organized for inference deployment with **FastAPI** backend and **Streamlit** frontend.

You can download the pre-trained model weights from this [Google Drive folder](https://drive.google.com/drive/folders/1brHks7I_8qJz80ypt4-6Hzcpkz0z79dI?usp=sharing) and place them in the `saved_models/` directory.

## Description
This project implements a music genre classification inference pipeline using FastAPI for the backend and Streamlit for the frontend. The application allows users to upload a `.wav` audio file, select a pre-trained model, and receive a predicted music genre along with class probabilities.  
The backend handles audio feature extraction using Librosa, loads pre-trained machine learning and deep learning models, and serves predictions through a FastAPI endpoint. The frontend provides an intuitive interface for users to interact with the application, upload audio files, and view results.

## 📁 Project Structure

```text
.
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                  # FastAPI application and /predict endpoint
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py                   # Streamlit frontend
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py                # Project paths, labels, and model registry
│   │   ├── feature_extraction.py    # Librosa feature extraction from first 3 seconds
│   │   ├── mlp.py                   # MLP architecture for deep learning inference
│   │   └── predictor.py             # Model loading and prediction logic
│   └── __init__.py
├── saved_models/
│   ├── hard_voting_default.joblib
│   ├── hard_voting_weight.joblib
│   ├── knn_default.joblib
│   ├── lgb_default.joblib
│   ├── soft_voting_default.joblib
│   ├── soft_voting_weight.joblib
│   ├── svm_default_probability.joblib
│   ├── xgb_default.joblib
│   └── mlp/
│       ├── mlp_state_dict.pt        # MLP state_dict saved by torch.save(model.state_dict(), ...)
│       └── scaler.joblib            # Scaler used for MLP features
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

## 📊 Benchmark Performance (Macro F1)

Accuracy values collected from notebook workflow.

| Model | Test Macro F1 |
| --- | ---: |
| KNN | 0.8778 |
| SVM (RBF) | 0.8511 |
| XGBoost | 0.8511 |
| LightGBM | 0.9070 |
| Hard Voting Ensemble Without Weights | 0.9136 |
| Hard Voting Ensemble With Weights | 0.9189 |
| Soft Voting Ensemble Without Weights | 0.9244 |
| Soft Voting Ensemble With Weights | 0.9264 |
| MLP (Deep Learning) | **0.9468** |

**Best Model**: MLP (Deep Learning)

## ⚙️ Installation

### 1. Install Dependencies

```bash
make install
```

Or manually:

```bash
pip install -r requirements.txt
```

## 🏃 Running the Application

### Option 1: Run API and UI Separately

Terminal 1:

```bash
make run-api
```

Terminal 2:

```bash
make run-ui
```

This starts:

- **FastAPI server**: http://127.0.0.1:8000
- **Streamlit UI**: http://localhost:8501

### Option 2: Run Manually

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
streamlit run src/ui/app.py
```

## 📝 Model Specifications

### Machine Learning Models

The machine learning models are saved as `.joblib` files in `saved_models/`.

Available models:

- `knn_default`
- `svm_default_probability`
- `xgb_default`
- `lgb_default`
- `hard_voting_default`
- `hard_voting_weight`
- `soft_voting_default`
- `soft_voting_weight`

Each machine learning model is expected to be saved as a full sklearn pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", model)
])
```

Therefore, the API does **not** apply a second scaler for machine learning models.

### Deep Learning Model

The deep learning model is an MLP loaded from `saved_models/mlp/`.

Expected files:

```text
saved_models/mlp/mlp_state_dict.pt
saved_models/mlp/scaler.joblib
```

The MLP model is loaded from a PyTorch `state_dict`:

```python
torch.save(model.state_dict(), "saved_models/mlp/mlp_state_dict.pt")
```

The MLP scaler is loaded separately and applied before inference.

### Audio Feature Extraction

- Input audio format: `.wav`
- Feature extraction library: `librosa`
- Only the **first 3 seconds** of audio are used
- Features are aligned with the columns from `data/features_3_sec.csv`
- Target labels are read from the dataset to preserve the same class order used during training

## 🎛️ Streamlit Model Selection Order

The UI displays models in this order:

```text
KNN
SVM
XGBoost
LightGBM
Hard Vote
Hard Vote Weighted
Soft Vote
Soft Vote Weighted
MLP
```

## 📋 API Endpoints

### GET `/health`

Check whether the API server is running.

**Response**:

```json
{
  "status": "ok"
}
```

### GET `/models`

List available models.

**Response**:

```json
{
  "models": [
    {
      "name": "knn_default",
      "display_name": "KNN",
      "type": "machine_learning"
    }
  ]
}
```

### POST `/predict`

Predict music genre from a `.wav` audio file.

**Request**: Multipart form data.

| Field | Type | Description |
| --- | --- | --- |
| `file` | File | `.wav` audio file |
| `model_name` | String | Model key, e.g. `knn_default`, `soft_voting_weight`, `mlp` |

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "model_name=mlp" \
  -F "file=@sample.wav"
```

**Response**:

```json
{
  "model_name": "mlp",
  "predicted_label": "rock",
  "probabilities": [
    {
      "label": "rock",
      "probability": 0.9123
    },
    {
      "label": "blues",
      "probability": 0.0312
    }
  ]
}
```

## ⚠️ Notes About Hard Voting

`VotingClassifier(voting="hard")` does not provide calibrated probabilities through `predict_proba()`.

For hard voting models, the API returns a vote-based probability distribution:

- `hard_voting_default`: probability is calculated from the ratio of votes from base estimators
- `hard_voting_weight`: probability is calculated from weighted votes

These values should be interpreted as **vote proportions**, not calibrated probabilities.

## 🧪 Quick Test

After starting the API server:

```bash
curl http://127.0.0.1:8000/health
```

Then test prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "model_name=soft_voting_weight" \
  -F "file=@your_audio.wav"
```
