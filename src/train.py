from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preparation import create_datasets
from model import build_baseline_cnn

EPOCHS = 10
DATASET_PATH = "data/raw"
MODEL_SAVE_PATH = "results/models/baseline_cnn.keras"
FIGURES_DIR = Path("results/figures")


def ensure_directories():
    Path("results/models").mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_training_curves(history):
    history_dict = history.history

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["accuracy"], label="Train Accuracy")
    plt.plot(history_dict["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history_dict["loss"], label="Train Loss")
    plt.plot(history_dict["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy curve saved to: {FIGURES_DIR / 'accuracy_curve.png'}")
    print(f"Loss curve saved to: {FIGURES_DIR / 'loss_curve.png'}")


def train_model():
    ensure_directories()

    train_dataset, val_dataset, test_dataset, class_names = create_datasets(DATASET_PATH)

    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    print(f"Class names: {class_names}")

    model = build_baseline_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
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
    print(f"Best model saved to: {MODEL_SAVE_PATH}")

    save_training_curves(history)

    test_results = model.evaluate(test_dataset, verbose=1)

    print("\nTEST RESULTS")
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {metric_value:.4f}")

    return history, model


if __name__ == "__main__":
    train_model()