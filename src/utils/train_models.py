from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.dl_model import MLPForClassification

SEED = 42
TEST_SIZE = 0.2


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_knn() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier())])


def build_svm() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("classifier", SVC(probability=True, random_state=SEED))])


def build_soft_voting() -> VotingClassifier:
    return VotingClassifier(
        estimators=[("knn", build_knn()), ("svm", build_svm())],
        voting="soft",
    )


def train_mlp(X_train: pd.DataFrame, y_train: pd.Series, labels: list[str], out_dir: Path, epochs: int = 80) -> None:
    mlp_dir = out_dir / "mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    x = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y_train.to_numpy(), dtype=torch.long)

    model = MLPForClassification(input_size=X_train.shape[1], num_classes=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    # Match your deployment format exactly: raw state_dict + separate scaler.
    torch.save(model.state_dict(), mlp_dir / "mlp_state_dict.pt")
    joblib.dump(scaler, mlp_dir / "scaler.joblib")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/features_3_sec.csv")
    parser.add_argument("--out-dir", default="saved_models")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    set_seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).drop_duplicates()
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])

    # Keep the same label order as the notebook: df['label'].unique(), not sorted().
    labels = df["label"].drop_duplicates().astype(str).tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    df["label"] = df["label"].map(label2id)

    feature_columns = [col for col in df.columns if col != "label"]
    X = df[feature_columns]
    y = df["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

    ml_models = {
        "knn_default": build_knn(),
        "svm_default_probability": build_svm(),
        "soft_voting_default": build_soft_voting(),
    }
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, out_dir / f"{name}.joblib")

    train_mlp(X_train, y_train, labels, out_dir=out_dir, epochs=args.epochs)

    metadata = {"labels": labels, "label2id": label2id, "feature_columns": feature_columns}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved models and metadata to {out_dir}")


if __name__ == "__main__":
    main()
