import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.utils.common import set_seed
from src.utils.visualize import eval


def build_models():
    knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    xgb_cls = xgb.XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )
    lgb_cls = lgb.LGBMClassifier(verbose=-1, random_state=42)

    hard_voting = VotingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("xgb", xgb_cls), ("lgb", lgb_cls)],
        voting="hard",
    )

    soft_voting = VotingClassifier(
        estimators=[("knn", knn), ("svm", svm), ("xgb", xgb_cls), ("lgb", lgb_cls)],
        voting="soft",
        weights=[0.905, 0.853, 0.897, 0.908],
    )

    return {
        "knn": knn,
        "svm_rbf": svm,
        "xgboost": xgb_cls,
        "lightgbm": lgb_cls,
        "hard_voting": hard_voting,
        "soft_voting": soft_voting,
    }


def train(data_csv: Path, output_dir: Path, test_size: float = 0.3, seed: int = 42) -> None:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])

    X = df.drop(columns=["label"])
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    metrics = {}
    best_name = None
    best_score = -1.0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)

        print(f"\n[{name}]")
        print(f"Train score: {train_score:.4f}")
        print(f"Test score:  {test_score:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics[name] = {
            "train_score": train_score,
            "test_score": test_score,
            "classification_report": report,
        }

        joblib.dump(model, output_dir / f"{name}.pkl")

        if test_score > best_score:
            best_score = test_score
            best_name = name

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(encoder, output_dir / "label_encoder.pkl")
    joblib.dump(X.columns.tolist(), output_dir / "feature_columns.pkl")

    assert best_name is not None
    best_model = joblib.load(output_dir / f"{best_name}.pkl")
    joblib.dump(best_model, output_dir / "best_model.pkl")

    metadata = {
        "best_model": best_name,
        "classes": encoder.classes_.tolist(),
        "feature_count": X.shape[1],
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    y_pred_best = best_model.predict(X_test_scaled)
    eval(y_test, y_pred_best, title=f"Best Model: {best_name}")

    print(f"\nSaved artifacts to: {output_dir}")
    print(f"Best model: {best_name} (test score={best_score:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train machine learning models and save artifacts")
    parser.add_argument("--data-csv", type=Path, default=Path("data/features_3_sec.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/machine_learning"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_csv, args.output_dir, args.test_size, args.seed)
