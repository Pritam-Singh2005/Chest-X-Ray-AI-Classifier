import os
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

# Pylance may report this as unresolved when TensorFlow is not installed in the
# selected interpreter; suppress the warning while keeping the runtime import.
from tensorflow.keras import layers, models  # type: ignore[import-not-found]
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(
    os.getenv(
        "CHEST_XRAY_DATASET",
        PROJECT_DIR / "dataset" / "chest_xray"
    )
)

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATASET_DIR}\n\n"
        "Set the CHEST_XRAY_DATASET environment variable "
        "to your dataset location."
    )

print("=" * 60)
print("Weighted CNN - Chest X-Ray Classification")
print("=" * 60)

print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset directory : {DATASET_DIR}")


# ============================================================
# LOAD DATASET
# ============================================================

train_dir = DATASET_DIR / "train"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

class_names = train_ds.class_names

print("\nClasses:")
print(class_names)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

class_counts = {}

for class_name in class_names:
    class_path = train_dir / class_name

    image_count = len(
        [
            file
            for file in class_path.iterdir()
            if file.is_file()
        ]
    )

    class_counts[class_name] = image_count


print("\nClass counts:")

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")


classes = list(range(len(class_names)))

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=[
        class_id
        for class_id, class_name in enumerate(class_names)
        for _ in range(class_counts[class_name])
    ]
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")

for class_id, weight in class_weights.items():
    print(f"{class_names[class_id]}: {weight:.4f}")


# ============================================================
# DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
    ],
    name="augmentation"
)


# ============================================================
# BUILD CNN
# ============================================================

model = models.Sequential(
    [
        layers.Input(
            shape=IMAGE_SIZE + (3,)
        ),

        augmentation,

        layers.Rescaling(
            1.0 / 255
        ),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ],
    name="Weighted_CNN"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)


# ============================================================
# SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

model_path = MODEL_DIR / "weighted_cnn.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting weighted CNN training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(model_path)

print("\nModel saved:")
print(model_path)


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Weighted CNN Training and Validation Accuracy"
)

plt.legend()
plt.tight_layout()

accuracy_path = RESULTS_DIR / "weighted_cnn_accuracy.png"

plt.savefig(accuracy_path)
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Weighted CNN Training and Validation Loss"
)

plt.legend()
plt.tight_layout()

loss_path = RESULTS_DIR / "weighted_cnn_loss.png"

plt.savefig(loss_path)
plt.close()


print("\nGraphs saved:")
print(accuracy_path)
print(loss_path)

print("\nWeighted CNN training completed.")