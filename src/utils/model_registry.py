from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "saved_models"
DEFAULT_DATA_CSV = PROJECT_ROOT / "data" / "features_3_sec.csv"

STATE_DICT_CANDIDATES = [
    "mlp_state_dict.pt",
    "model_state_dict.pt",
    "state_dict.pt",
    "pytorch_model.bin",
]
SCALER_CANDIDATES = [
    "mlp_scaler.joblib",
    "scaler.joblib",
    "standard_scaler.joblib",
    "mlp_scaler.pkl",
    "scaler.pkl",
]

MODEL_ORDER = [
    "knn_default",
    "svm_default_probability",
    "xgb_default",
    "lgb_default",
    "hard_voting_default",
    "hard_voting_weight",
    "soft_voting_default",
    "soft_voting_weight",
]

DISPLAY_NAMES = {
    "knn_default": "KNN",
    "svm_default_probability": "SVM",
    "xgb_default": "XGBoost",
    "lgb_default": "LightGBM",
    "hard_voting_default": "Hard Vote",
    "hard_voting_weight": "Hard Vote Weighted",
    "soft_voting_default": "Soft Vote",
    "soft_voting_weight": "Soft Vote Weighted",
    "mlp": "MLP",
}


def _as_path(path: str | Path | None, default: Path) -> Path:
    return Path(path) if path is not None else default


def _labels_and_features_from_csv(csv_path: str | Path = DEFAULT_DATA_CSV) -> dict[str, Any]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing metadata and CSV. Please keep data/features_3_sec.csv or create saved_models/metadata.json."
        )

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("data/features_3_sec.csv must contain a 'label' column.")

    if "filename" in df.columns:
        df = df.drop(columns=["filename"])

    # Mirrors the notebook: classes = df['label'].unique(). Do not sort labels.
    labels = df["label"].drop_duplicates().astype(str).tolist()
    label2id = {label: idx for idx, label in enumerate(labels)}
    feature_columns = [col for col in df.columns if col != "label"]

    return {
        "labels": labels,
        "label2id": label2id,
        "feature_columns": feature_columns,
    }


def load_metadata(
    models_dir: str | Path | None = None,
    csv_path: str | Path = DEFAULT_DATA_CSV,
) -> dict[str, Any]:
    """Load metadata, falling back to features_3_sec.csv when metadata.json is absent."""
    root = _as_path(models_dir, DEFAULT_MODELS_DIR)
    path = root / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return _labels_and_features_from_csv(csv_path)


def _find_first_existing(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        direct = root / name
        nested = root / "mlp" / name
        if direct.exists():
            return direct
        if nested.exists():
            return nested
    return None


def _ordered_model_names(names: list[str]) -> list[str]:
    known = [name for name in MODEL_ORDER if name in names]
    unknown = sorted(name for name in names if name not in MODEL_ORDER)
    return known + unknown


def list_models(models_dir: str | Path | None = None) -> dict[str, list[str]]:
    root = _as_path(models_dir, DEFAULT_MODELS_DIR)
    ml_names = [p.stem for p in root.glob("*.joblib") if "scaler" not in p.stem.lower()]
    ml = _ordered_model_names(ml_names)

    dl: list[str] = []
    if _find_first_existing(root, STATE_DICT_CANDIDATES) is not None:
        dl.append("mlp")
    elif (root / "mlp").is_dir() and any((root / "mlp").glob("*.pt")):
        dl.append("mlp")

    return {"machine_learning": ml, "deep_learning": dl}


def list_model_options(models_dir: str | Path | None = None) -> dict[str, list[dict[str, str]]]:
    names = list_models(models_dir)
    return {
        group: [{"name": name, "display_name": DISPLAY_NAMES.get(name, name)} for name in group_names]
        for group, group_names in names.items()
    }


def _align_to_label_order(probabilities: np.ndarray, classes: Any, n_labels: int) -> np.ndarray:
    """Return probability vector in label-id order 0..n_labels-1.

    sklearn exposes probabilities in model.classes_ order. In this project labels are
    encoded as 0..9 in the order read from CSV, so API responses should use that order.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if classes is None:
        return probabilities

    aligned = np.zeros(n_labels, dtype=float)
    for prob, cls in zip(probabilities, classes):
        idx = int(cls)
        if 0 <= idx < n_labels:
            aligned[idx] = float(prob)
    return aligned


def _predict_hard_vote_probabilities(model: Any, features: pd.DataFrame, n_labels: int) -> np.ndarray:
    """Approximate probabilities for sklearn VotingClassifier(voting='hard').

    Hard voting classifiers intentionally do not implement predict_proba(). For UI
    probability tables, we convert the fitted estimators' class votes into a normalized
    vote distribution. If weights were used during training, the weighted vote mass is
    used, matching sklearn's hard voting behavior.
    """
    classes = list(getattr(model, "classes_", range(n_labels)))
    class_to_pos = {int(cls): pos for pos, cls in enumerate(classes)}
    vote_mass = np.zeros(len(classes), dtype=float)

    estimators = list(getattr(model, "estimators_", []))
    if not estimators:
        pred = int(model.predict(features)[0])
        if pred in class_to_pos:
            vote_mass[class_to_pos[pred]] = 1.0
    else:
        weights = getattr(model, "weights", None)
        if weights is None:
            weights = [1.0] * len(estimators)

        for estimator, weight in zip(estimators, weights):
            pred = int(estimator.predict(features)[0])
            if pred in class_to_pos:
                vote_mass[class_to_pos[pred]] += float(weight)

    total = vote_mass.sum()
    if total <= 0:
        raise ValueError("Hard voting model produced no valid votes.")

    probabilities_in_model_class_order = vote_mass / total
    return _align_to_label_order(probabilities_in_model_class_order, classes, n_labels)


def predict_ml(
    model_name: str,
    features: pd.DataFrame,
    models_dir: str | Path | None = None,
    n_labels: int | None = None,
) -> np.ndarray:
    root = _as_path(models_dir, DEFAULT_MODELS_DIR)
    model_path = root / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"ML model not found: {model_path}")

    # Your ML files are sklearn Pipelines, already containing scaler + classifier.
    model = joblib.load(model_path)
    n_labels = n_labels or len(load_metadata(root)["labels"])

    voting = getattr(model, "voting", None)
    if voting == "hard":
        return _predict_hard_vote_probabilities(model, features, n_labels)

    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model {model_name} does not support predict_proba().")

    probabilities = model.predict_proba(features)[0]
    classes = getattr(model, "classes_", None)
    return _align_to_label_order(probabilities, classes, n_labels)


def _load_scaler(root: Path) -> Any:
    scaler_path = _find_first_existing(root, SCALER_CANDIDATES)
    if scaler_path is None:
        raise FileNotFoundError(
            "MLP scaler not found. Expected one of: "
            + ", ".join([f"saved_models/{name}" for name in SCALER_CANDIDATES])
            + " or the same filenames under saved_models/mlp/."
        )
    return joblib.load(scaler_path)


def _load_state_dict(root: Path) -> dict[str, Any]:
    import torch

    state_path = _find_first_existing(root, STATE_DICT_CANDIDATES)
    if state_path is None:
        raise FileNotFoundError(
            "MLP state_dict not found. Expected saved_models/mlp_state_dict.pt "
            "or saved_models/mlp/mlp_state_dict.pt."
        )
    return torch.load(state_path, map_location="cpu")


def predict_dl(
    model_name: str,
    features: pd.DataFrame,
    models_dir: str | Path | None = None,
    labels: list[str] | None = None,
) -> np.ndarray:
    if model_name != "mlp":
        raise ValueError("Only deep learning model name currently supported is 'mlp'.")

    import torch

    from src.utils.dl_model import MLPForClassification

    root = _as_path(models_dir, DEFAULT_MODELS_DIR)
    labels = labels or load_metadata(root)["labels"]

    scaler = _load_scaler(root)
    state_dict = _load_state_dict(root)

    # torch.save(model.state_dict(), ...) saves only the raw parameter dictionary.
    # Some workflows save {'state_dict': ...}; support both, but prefer the raw form.
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = MLPForClassification(input_size=features.shape[1], num_classes=len(labels))
    model.load_state_dict(state_dict)
    model.eval()

    x = scaler.transform(features)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return np.asarray(probabilities, dtype=float)
