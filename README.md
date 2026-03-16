# Music Genre Classification Inference Project

Project structure to train models from GTZAN `features_3_sec.csv` and run inference with FastAPI/Streamlit. Using ChatGPT for code generation, refactoring, and documentation.

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

Equivalent command:

```bash
python -m src.training.train_machine_learning --data-csv data/features_3_sec.csv --output-dir models/machine_learning
```

Train deep learning (MLP):

```bash
make train-dl
```

Equivalent command:

```bash
python -m src.training.train_deep_learning --data-csv data/features_3_sec.csv --output-dir models/deep_learning
```

Generated ML artifacts (`models/machine_learning`):

- `knn.pkl`
- `svm_rbf.pkl`
- `xgboost.pkl`
- `lightgbm.pkl`
- `hard_voting.pkl`
- `soft_voting.pkl`
- `best_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `feature_columns.pkl`
- `metrics.json`
- `metadata.json`

Generated DL artifacts (`models/deep_learning`):

- `mlp_checkpoint.pth` (contains: `model_state_dict`, `label2id`, `id2label`)
- `scaler.pkl`
- `feature_columns.pkl`
- `metadata.json`

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
