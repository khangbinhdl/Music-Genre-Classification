# 🎵 Music Genre Classification Inference Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S9P2BcgoNqyvvmIidDVVj5nxITxpiQA-?usp=sharing)

Train models from GTZAN `features_3_sec.csv` to predict music genres from 3-second audio features.

This repository is a production-style refactor of the original Colab notebook above, with GitHub Copilot support for code organization, refactoring, and building inference apps (FastAPI + Streamlit).

## 📁 Project Structure

```
.
├── src/                          # Source code
│   ├── inference/                # Model inference
│   │   ├── __init__.py
│   │   ├── feature_extraction.py # Audio feature extraction
│   │   └── predictor.py         # Predictor class
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── mlp.py               # Deep learning model
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   ├── train_machine_learning.py
│   │   ├── train_deep_learning.py
│   │   └── train_models.py
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── common.py            # Common utilities
│   │   └── visualize.py         # Visualization & evaluation
│   └── __init__.py
├── ui/                           # Streamlit frontend
│   └── streamlit_app.py         # UI application
├── api/                          # FastAPI backend
│   └── fastapi_app.py           # API endpoints
├── notebook/                     # Jupyter notebooks
├── data/                         # Dataset files
│   ├── features_3_sec.csv
│   └── features_30_sec.csv
├── models/                       # Pre-trained model weights
│   ├── machine_learning/
│   └── deep_learning/
├── requirements.txt              # Python dependencies
├── Makefile                      # Commands for easy execution
└── README.md                     # This file
```

## 📊 Benchmark Performance (Accuracy)

Accuracy values collected from notebook workflow and saved artifacts:

- Machine learning models: `models/machine_learning/metrics.json`
- Deep learning model (MLP): `models/deep_learning/metadata.json`

| Model | Test Accuracy |
| --- | ---: |
| KNN | 0.8969 |
| SVM (RBF) | 0.8415 |
| XGBoost | 0.8959 |
| LightGBM | 0.8992 |
| Hard Voting Ensemble | 0.9139 |
| Soft Voting Ensemble | 0.9243 |
| MLP (Deep Learning) | **0.9313** |

**Best Model**: MLP (Deep Learning)

*Note: Results may differ from original Colab notebook due to re-running on MacBook with Apple Silicon MPS.*

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

### Option 1: Run Both Servers (Recommended)

```bash
make run
```

This starts:
- **API Server**: http://127.0.0.1:8000
- **Streamlit UI**: http://localhost:8501

### Option 2: Run Separately

```bash
# Terminal 1: Start FastAPI
make run-fastapi

# Terminal 2: Start Streamlit
make run-streamlit
```

## 📚 Training

### Train Machine Learning Models

```bash
make train-ml
```

### Train Deep Learning Model (MLP)

```bash
make train-dl
```

## 🎯 Model Selection

In Streamlit UI, choose your preferred model:
- **Best Model (Machine Learning)**: Soft Voting Ensemble (0.9243 accuracy)
- **MLP (Deep Learning)**: Neural Network (0.9313 accuracy)

## 📝 Model Specifications

### Machine Learning Models

- KNN, SVM (RBF), XGBoost, LightGBM
- Ensemble voting (hard & soft)
- Input: 3-second audio features (57 features)

### Deep Learning (MLP)

- Architecture: Multi-Layer Perceptron
- Input dimension: 57 audio features
- Max sequence length: 3 seconds
- Output: 10 genre classes

## 📋 API Endpoints

### POST `/predict`

Predict genre from audio file.

**Request**: Multipart form with `file` field containing `.wav` audio

**Response**:
```json
{
  "genre": "string",
  "confidence": 0.95
}
```
