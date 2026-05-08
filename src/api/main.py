from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.utils.audio_features import extract_features_from_wav
from src.utils.model_registry import DEFAULT_MODELS_DIR, list_model_options, list_models, load_metadata, predict_dl, predict_ml

MODELS_DIR = DEFAULT_MODELS_DIR

app = FastAPI(title="GTZAN Music Genre Classification API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict[str, object]:
    try:
        metadata = load_metadata(MODELS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"models": list_models(MODELS_DIR), "model_options": list_model_options(MODELS_DIR), "labels": metadata["labels"]}


@app.post("/predict")
async def predict(
    model_type: str = Form(..., description="machine_learning or deep_learning"),
    model_name: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, object]:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file.")

    tmp_path: Path | None = None
    try:
        metadata = load_metadata(MODELS_DIR)
        feature_columns = metadata["feature_columns"]
        labels = metadata["labels"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        features = extract_features_from_wav(tmp_path, duration=3.0, feature_columns=feature_columns)

        if model_type == "machine_learning":
            probabilities = predict_ml(model_name, features, MODELS_DIR, n_labels=len(labels))
        elif model_type == "deep_learning":
            probabilities = predict_dl(model_name, features, MODELS_DIR, labels=labels)
        else:
            raise HTTPException(status_code=400, detail="model_type must be machine_learning or deep_learning.")

        if len(probabilities) != len(labels):
            raise ValueError(
                f"Number of probabilities ({len(probabilities)}) does not match number of labels ({len(labels)})."
            )

        table = pd.DataFrame({"genre": labels, "probability": probabilities})
        table = table.sort_values("probability", ascending=False)
        table["probability"] = table["probability"].round(6)

        return {
            "filename": file.filename,
            "model_type": model_type,
            "model_name": model_name,
            "top_prediction": table.iloc[0].to_dict(),
            "probabilities": table.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
