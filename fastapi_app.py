from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.predictor import MusicGenrePredictor

app = FastAPI(title="Music Genre Inference API", version="1.0.0")
predictor = MusicGenrePredictor(model_dir="models/machine_learning")


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "Music genre API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is empty")
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = predictor.predict_from_bytes(audio_bytes)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
