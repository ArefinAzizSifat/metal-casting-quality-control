from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
SEED = 42


def is_image_file(file_path):
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS


def get_class_folders(dataset_path):
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    class_folders = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

    if not class_folders:
        raise ValueError(f"No class folders found inside: {dataset_path}")

    return sorted(class_folders)


def get_image_files(folder_path):
    folder_path = Path(folder_path)
    return sorted([file for file in folder_path.rglob("*") if is_image_file(file)])


def inspect_dataset(dataset_path):
    class_folders = get_class_folders(dataset_path)

    class_counts = {}
    image_sizes = []
    image_modes = []
    corrupted_images = []
    sample_images = {}

    for class_folder in class_folders:
        image_files = get_image_files(class_folder)
        class_counts[class_folder.name] = len(image_files)
        sample_images[class_folder.name] = [str(path) for path in image_files[:5]]

        for image_path in image_files:
            try:
                with Image.open(image_path) as img:
                    img.verify()

                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
                    image_modes.append(img.mode)

            except (UnidentifiedImageError, OSError, ValueError):
                corrupted_images.append(str(image_path))

    summary = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "classes": [folder.name for folder in class_folders],
        "class_counts": class_counts,
        "total_images": sum(class_counts.values()),
        "image_size_distribution": dict(Counter(image_sizes).most_common(10)),
        "image_mode_distribution": dict(Counter(image_modes)),
        "corrupted_images": corrupted_images,
        "sample_images": sample_images,
    }

    return summary


def print_dataset_summary(summary):
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Dataset path: {summary['dataset_path']}")
    print(f"Classes found: {summary['classes']}")
    print(f"Total images: {summary['total_images']}")
    print()

    print("Image count per class:")
    for class_name, count in summary["class_counts"].items():
        print(f"  - {class_name}: {count}")
    print()

    print("Most common image sizes:")
    if summary["image_size_distribution"]:
        for size, count in summary["image_size_distribution"].items():
            print(f"  - {size}: {count}")
    else:
        print("  - No readable images found.")
    print()

    print("Image mode distribution:")
    if summary["image_mode_distribution"]:
        for mode, count in summary["image_mode_distribution"].items():
            print(f"  - {mode}: {count}")
    else:
        print("  - No image modes found.")
    print()

    print(f"Corrupted images found: {len(summary['corrupted_images'])}")
    if summary["corrupted_images"]:
        for image_path in summary["corrupted_images"][:10]:
            print(f"  - {image_path}")
    print()

    print("Sample image paths per class:")
    for class_name, image_paths in summary["sample_images"].items():
        print(f"  - {class_name}:")
        if image_paths:
            for image_path in image_paths:
                print(f"      {image_path}")
        else:
            print("      No images found.")
    print("=" * 60)


def create_datasets(dataset_path):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True
    )

    temp_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=True
    )

    temp_batches = tf.data.experimental.cardinality(temp_dataset).numpy()
    val_batches = int(0.5 * temp_batches)

    val_dataset = temp_dataset.take(val_batches)
    test_dataset = temp_dataset.skip(val_batches)

    class_names = train_dataset.class_names

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    val_dataset = val_dataset.prefetch(buffer_size=autotune)
    test_dataset = test_dataset.prefetch(buffer_size=autotune)

    return train_dataset, val_dataset, test_dataset, class_names


def print_dataset_split_info(train_dataset, val_dataset, test_dataset, class_names):
    print("\n" + "=" * 60)
    print("DATASET SPLIT INFO")
    print("=" * 60)
    print(f"Class names: {class_names}")
    print(f"Training batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation batches: {tf.data.experimental.cardinality(val_dataset).numpy()}")
    print(f"Test batches: {tf.data.experimental.cardinality(test_dataset).numpy()}")
    print(f"Image size used for training: ({IMG_HEIGHT}, {IMG_WIDTH})")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)


if __name__ == "__main__":
    DATASET_PATH = "data/raw"

    summary = inspect_dataset(DATASET_PATH)
    print_dataset_summary(summary)

    train_dataset, val_dataset, test_dataset, class_names = create_datasets(DATASET_PATH)
    print_dataset_split_info(train_dataset, val_dataset, test_dataset, class_names)