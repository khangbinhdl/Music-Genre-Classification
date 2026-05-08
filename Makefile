PYTHON ?= python
PIP ?= pip
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
UI_PORT ?= 8501

install:
	$(PIP) install -r requirements.txt

run-api:
	uvicorn src.api.main:app --host $(API_HOST) --port $(API_PORT) --reload

run-ui:
	streamlit run src/ui/app.py --server.port $(UI_PORT)

# Optional helper only if you want to retrain from data/features_3_sec.csv.
train:
	$(PYTHON) -m src.utils.train_models --csv data/features_3_sec.csv --out-dir saved_models
