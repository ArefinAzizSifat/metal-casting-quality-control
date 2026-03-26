from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preparation import create_datasets
from model import build_baseline_cnn, build_improved_cnn

MODEL_TYPE = "improved"   # "baseline" or "improved"
EPOCHS = 15
DATASET_PATH = "data/raw"

MODEL_SAVE_PATHS = {
    "baseline": "results/models/baseline_cnn.keras",
    "improved": "results/models/improved_cnn.keras",
}

FIGURE_PREFIX = {
    "baseline": "baseline",
    "improved": "improved",
}

FIGURES_DIR = Path("results/figures")


def ensure_directories():
    Path("results/models").mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_model(model_type):
    if model_type == "baseline":
        return build_baseline_cnn()
    if model_type == "improved":
        return build_improved_cnn()
    raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")


def save_training_curves(history, model_type):
    history_dict = history.history
    prefix = FIGURE_PREFIX[model_type]

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["accuracy"], label="Train Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_type.capitalize()} CNN: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    accuracy_path = FIGURES_DIR / f"{prefix}_accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title(f"{model_type.capitalize()} CNN: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = FIGURES_DIR / f"{prefix}_loss_curve.png"
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy curve saved to: {accuracy_path}")
    print(f"Loss curve saved to: {loss_path}")


def train_model():
    ensure_directories()

    train_dataset, val_dataset, test_dataset, class_names = create_datasets(DATASET_PATH)
    model_save_path = MODEL_SAVE_PATHS[MODEL_TYPE]

    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Class names: {class_names}")

    model = get_model(MODEL_TYPE)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print(f"Best model saved to: {model_save_path}")

    save_training_curves(history, MODEL_TYPE)

    test_results = model.evaluate(test_dataset, verbose=1)

    print("\nTEST RESULTS")
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {metric_value:.4f}")

    return history, model


if __name__ == "__main__":
    train_model()