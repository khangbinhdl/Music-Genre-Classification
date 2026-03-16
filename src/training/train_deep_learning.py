import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import MLP
from src.utils.common import set_seed
from src.utils.visualize import eval


def calculate_accuracy(loader: DataLoader, model: nn.Module, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total else 0.0


def train(
    data_csv: Path,
    output_dir: Path,
    seed: int = 42,
    test_size: float = 0.3,
    batch_size: int = 256,
    num_epochs: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
) -> None:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])

    X = df.drop(columns=["label"])
    y = df["label"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    label2id = {label: int(i) for i, label in enumerate(encoder.classes_)}
    id2label = {int(i): label for i, label in enumerate(encoder.classes_)}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X_train.shape[1]
    num_classes = len(encoder.classes_)

    model = MLP(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs.float())
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}, Train Loss: {loss.item():.4f}")
            step += 1

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Accuracy: {100 * correct / total:.2f}%, LR: {current_lr:.6f}"
            )

    train_accuracy = calculate_accuracy(train_loader, model, device)
    test_accuracy = calculate_accuracy(val_loader, model, device)
    print("Training set score: {:.3f}".format(train_accuracy))
    print("Test set score: {:.3f}".format(test_accuracy))

    model.eval()
    y_preds = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            _, predicted = outputs.max(1)
            y_preds.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    eval(y_true, y_preds, title="MLP Confusion Matrix")
    print("\nMLP Classification Report:")
    print(classification_report(y_true, y_preds, target_names=[id2label[i] for i in range(num_classes)]))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
        },
        output_dir / "mlp_checkpoint.pth",
    )

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(X.columns.tolist(), output_dir / "feature_columns.pkl")

    metadata = {
        "model": "mlp",
        "input_size": int(input_size),
        "num_classes": int(num_classes),
        "classes": encoder.classes_.tolist(),
        "train_score": float(train_accuracy),
        "test_score": float(test_accuracy),
        "device": device,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved MLP artifacts to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train deep learning MLP model and save artifacts")
    parser.add_argument("--data-csv", type=Path, default=Path("data/features_3_sec.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/deep_learning"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        seed=args.seed,
        test_size=args.test_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
