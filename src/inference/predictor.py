import tempfile
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from src.inference.feature_extraction import ProgressCallback, extract_features_frame


class MusicGenrePredictor:
    def __init__(self, model_dir: str = "models/machine_learning"):
        model_path = Path(model_dir)
        self.model = joblib.load(model_path / "best_model.pkl")
        self.scaler = joblib.load(model_path / "scaler.pkl")
        self.encoder = joblib.load(model_path / "label_encoder.pkl")
        self.feature_columns = joblib.load(model_path / "feature_columns.pkl")

    def predict_from_wav_path(
        self,
        wav_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        use_tqdm: bool = False,
    ) -> Dict[str, object]:
        features_df = extract_features_frame(
            wav_path,
            self.feature_columns,
            progress_callback=progress_callback,
            use_tqdm=use_tqdm,
        )
        features_scaled = self.scaler.transform(features_df)

        pred_id = int(self.model.predict(features_scaled)[0])
        pred_label = str(self.encoder.inverse_transform([pred_id])[0])

        result = {
            "predicted_id": pred_id,
            "predicted_genre": pred_label,
        }

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_scaled)[0]
            labels = self.encoder.classes_
            result["probabilities"] = {
                str(label): float(prob) for label, prob in zip(labels, probs)
            }

        return result

    def predict_from_bytes(
        self,
        audio_bytes: bytes,
        progress_callback: Optional[ProgressCallback] = None,
        use_tqdm: bool = False,
    ) -> Dict[str, object]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.predict_from_wav_path(
                tmp.name,
                progress_callback=progress_callback,
                use_tqdm=use_tqdm,
            )


class MLPGenrePredictor:
    def __init__(self, model_dir: str = "models/deep_learning"):
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from src.models import MLP  # pylint: disable=import-outside-toplevel
        except Exception as exc:
            raise RuntimeError(
                "MLP inference requires torch. Install dependencies and use the correct environment."
            ) from exc

        self._torch = torch
        model_path = Path(model_dir)
        self.device = "cpu"

        checkpoint = self._torch.load(model_path / "mlp_checkpoint.pth", map_location=self.device)
        self.label2id = checkpoint["label2id"]
        self.id2label = {int(k): v for k, v in checkpoint["id2label"].items()}

        self.scaler = joblib.load(model_path / "scaler.pkl")
        self.feature_columns = joblib.load(model_path / "feature_columns.pkl")

        input_size = len(self.feature_columns)
        num_classes = len(self.label2id)
        self.model = MLP(input_size=input_size, num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_from_wav_path(
        self,
        wav_path: str,
        progress_callback: Optional[ProgressCallback] = None,
        use_tqdm: bool = False,
    ) -> Dict[str, object]:
        features_df = extract_features_frame(
            wav_path,
            self.feature_columns,
            progress_callback=progress_callback,
            use_tqdm=use_tqdm,
        )
        features_scaled = self.scaler.transform(features_df)

        x = self._torch.tensor(features_scaled, dtype=self._torch.float32).to(self.device)
        with self._torch.no_grad():
            logits = self.model(x)
            probs = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = str(self.id2label[pred_id])

        labels_sorted = [self.id2label[i] for i in range(len(probs))]
        return {
            "predicted_id": pred_id,
            "predicted_genre": pred_label,
            "probabilities": {
                str(label): float(prob) for label, prob in zip(labels_sorted, probs)
            },
        }

    def predict_from_bytes(
        self,
        audio_bytes: bytes,
        progress_callback: Optional[ProgressCallback] = None,
        use_tqdm: bool = False,
    ) -> Dict[str, object]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            return self.predict_from_wav_path(
                tmp.name,
                progress_callback=progress_callback,
                use_tqdm=use_tqdm,
            )
