# Music Genre Classification Inference Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S9P2BcgoNqyvvmIidDVVj5nxITxpiQA-?usp=sharing)

Project structure to train models from GTZAN `features_3_sec.csv` to predict music genres from 3-second audio features. 

This repository is a production-style refactor of the original Colab notebook above, with GitHub Copilot support for code organization, refactoring, and building inference apps (FastAPI + Streamlit).

## Benchmark Performance (Accuracy)

Accuracy values below are collected from the notebook workflow and saved artifacts:

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

Best observed accuracy in this benchmark: **MLP (Deep Learning)**.

*Note: These results may differ from the original Colab notebook because training/inference was re-run on MacBook using Apple Silicon MPS.*

## Structure

- `src/utils/common.py`: `set_seed` utility.
- `src/utils/visualize.py`: `eval` function (confusion matrix + classification report).
- `src/training/train_machine_learning.py`: train ML models and save artifacts.
- `src/training/train_deep_learning.py`: train MLP model and save artifacts.
- `src/training/train_models.py`: backward-compatible entrypoint for ML training.
- `src/inference/feature_extraction.py`: extract 3-second audio features from `.wav` using `librosa`.
- `src/inference/predictor.py`: shared predictor class for API and UI.
- `fastapi_app.py`: inference API.
- `streamlit_app.py`: Streamlit upload UI.
- `models/`: output folder for artifacts.

## Install

```bash
pip install -r requirements.txt
```

## Train and Save PKL

```bash
make train-ml
```

Train deep learning (MLP):

```bash
make train-dl
```

## Run FastAPI

```bash
make run-fastapi
```

Test endpoint: `POST /predict` with multipart field name `file`.

## Run Streamlit

```bash
make run-streamlit
```

In Streamlit, you can choose inference model: `Best Model (Machine Learning)` or `MLP (Deep Learning)`.
