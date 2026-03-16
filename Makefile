PYTHON ?= python
UVICORN ?= uvicorn
STREAMLIT ?= streamlit

DATA_CSV ?= data/features_3_sec.csv
ML_MODEL_DIR ?= models/machine_learning
DL_MODEL_DIR ?= models/deep_learning

.PHONY: train train-ml train-dl run-fastapi run-streamlit

train:
	$(MAKE) train-ml

train-ml:
	$(PYTHON) -m src.training.train_machine_learning --data-csv "$(DATA_CSV)" --output-dir "$(ML_MODEL_DIR)"

train-dl:
	$(PYTHON) -m src.training.train_deep_learning --data-csv "$(DATA_CSV)" --output-dir "$(DL_MODEL_DIR)"

run-fastapi:
	$(PYTHON) -m uvicorn fastapi_app:app --reload

run-streamlit:
	$(PYTHON) -m streamlit run streamlit_app.py
