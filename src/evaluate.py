from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from data_preparation import create_datasets

DATASET_PATH = "data/raw"
MODEL_PATH = "results/models/baseline_cnn.keras"
FIGURES_DIR = Path("results/figures")


def ensure_paths():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_figure(cm, class_names):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix figure saved to: {save_path}")


def save_prediction_distribution_figure(y_pred_prob):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_pred_prob, bins=20)
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    save_path = FIGURES_DIR / "prediction_probability_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Prediction probability figure saved to: {save_path}")


def evaluate_model():
    ensure_paths()

    _, _, test_dataset, class_names = create_datasets(DATASET_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred_prob = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_prob.extend(predictions.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true).astype(int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Class names: {class_names}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()

    print("Confusion Matrix:")
    print(cm)
    print()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("=" * 60)

    save_confusion_matrix_figure(cm, class_names)
    save_prediction_distribution_figure(y_pred_prob)


if __name__ == "__main__":
    evaluate_model()