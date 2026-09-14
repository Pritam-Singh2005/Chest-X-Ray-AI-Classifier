import os
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt

# Pyright can emit a false-positive missing-source warning for TensorFlow.
# This project expects the TensorFlow package to be installed in the active env.
# pyright: reportMissingModuleSource=false
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


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

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

TRANSFER_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_transfer.keras"
)

FINETUNED_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_finetuned.keras"
)


# ============================================================
# CHECK FILES
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATASET_DIR}"
    )

if not TRANSFER_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nTransfer learning model not found:\n"
        f"{TRANSFER_MODEL_PATH}\n\n"
        "Run mobilenetv2_transfer.py first."
    )


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("MobileNetV2 Fine-Tuning")
print("=" * 60)

print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset directory : {DATASET_DIR}")
print(f"Input size        : {IMAGE_SIZE}")
print(f"Transfer model    : {TRANSFER_MODEL_PATH}")


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


# ============================================================
# OPTIMIZE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# LOAD TRANSFER LEARNING MODEL
# ============================================================

print("\nLoading transfer learning model...")

model = load_model(
    TRANSFER_MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# FIND MOBILENETV2 BASE MODEL
# ============================================================

base_model = None

for layer in model.layers:

    if (
        "mobilenetv2" in layer.name.lower()
        or "mobilenet" in layer.name.lower()
    ):
        if hasattr(layer, "layers"):
            base_model = layer
            break


if base_model is None:
    raise RuntimeError(
        "MobileNetV2 base model could not be found."
    )


print("\nBase model found:")
print(base_model.name)


# ============================================================
# UNFREEZE LAST 30 LAYERS
# ============================================================

base_model.trainable = True

fine_tune_from = max(
    0,
    len(base_model.layers) - 30
)

for index, layer in enumerate(base_model.layers):

    if index < fine_tune_from:
        layer.trainable = False

    else:
        layer.trainable = True

    # Batch Normalization layers remain frozen
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


# ============================================================
# SHOW TRAINABLE LAYERS
# ============================================================

trainable_count = sum(
    1
    for layer in model.layers
    if layer.trainable
)

print(
    f"\nTrainable top-level layers: "
    f"{trainable_count}"
)


# ============================================================
# RECOMPILE
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=FINETUNED_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]


# ============================================================
# FINE-TUNE MODEL
# ============================================================

print("\nStarting fine-tuning...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    FINETUNED_MODEL_PATH
)

print("\nFine-tuned model saved:")
print(FINETUNED_MODEL_PATH)


# ============================================================
# ACCURACY GRAPH
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
    "MobileNetV2 Fine-Tuning Accuracy"
)

plt.legend()
plt.tight_layout()

accuracy_path = (
    RESULTS_DIR /
    "mobilenetv2_finetuning_accuracy.png"
)

plt.savefig(accuracy_path)
plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

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
    "MobileNetV2 Fine-Tuning Loss"
)

plt.legend()
plt.tight_layout()

loss_path = (
    RESULTS_DIR /
    "mobilenetv2_finetuning_loss.png"
)

plt.savefig(loss_path)
plt.close()


print("\nTraining graphs saved:")
print(accuracy_path)
print(loss_path)

print("\nFine-tuning completed successfully.")