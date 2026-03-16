from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def eval(y_true, y_pred, title: Optional[str] = None) -> str:
    """Visualize confusion matrix and print classification report."""
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.set(rc={"figure.figsize": (12, 6)})
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")

    if title:
        plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    report = classification_report(y_true, y_pred)
    print(report)
    return report
