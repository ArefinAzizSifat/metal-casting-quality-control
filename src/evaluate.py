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

MODEL_TYPE = "improved"   # "baseline" or "improved"
DATASET_PATH = "data/raw"
FIGURES_DIR = Path("results/figures")

MODEL_PATHS = {
    "baseline": "results/models/baseline_cnn.keras",
    "improved": "results/models/improved_cnn.keras",
}

FILE_PREFIX = {
    "baseline": "baseline",
    "improved": "improved",
}


def ensure_paths(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix_figure(cm, class_names, model_type):
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"{model_type.capitalize()} CNN - Confusion Matrix")
    plt.tight_layout()

    save_path = FIGURES_DIR / f"{FILE_PREFIX[model_type]}_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix figure saved to: {save_path}")


def save_prediction_distribution_figure(y_pred_prob, model_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_pred_prob, bins=20)
    ax.set_title(f"{model_type.capitalize()} CNN - Prediction Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    save_path = FIGURES_DIR / f"{FILE_PREFIX[model_type]}_prediction_probability_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Prediction probability figure saved to: {save_path}")


def save_classification_report_file(acc, prec, rec, f1, cm, report_text, class_names, model_type):
    save_path = FIGURES_DIR / f"{FILE_PREFIX[model_type]}_classification_report.txt"

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(f"{model_type.upper()} CNN EVALUATION RESULTS\n")
        file.write("=" * 60 + "\n")
        file.write(f"Class names: {class_names}\n")
        file.write(f"Accuracy : {acc:.4f}\n")
        file.write(f"Precision: {prec:.4f}\n")
        file.write(f"Recall   : {rec:.4f}\n")
        file.write(f"F1-score : {f1:.4f}\n\n")

        file.write("Confusion Matrix:\n")
        file.write(f"{cm}\n\n")

        file.write("Classification Report:\n")
        file.write(report_text)
        file.write("\n")

    print(f"Classification report saved to: {save_path}")


def evaluate_model():
    model_path = MODEL_PATHS[MODEL_TYPE]
    ensure_paths(model_path)

    _, _, test_dataset, class_names = create_datasets(DATASET_PATH)
    model = tf.keras.models.load_model(model_path)

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
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Model path: {model_path}")
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
    print(report_text)
    print("=" * 60)

    save_confusion_matrix_figure(cm, class_names, MODEL_TYPE)
    save_prediction_distribution_figure(y_pred_prob, MODEL_TYPE)
    save_classification_report_file(
        acc, prec, rec, f1, cm, report_text, class_names, MODEL_TYPE
    )


if __name__ == "__main__":
    evaluate_model()