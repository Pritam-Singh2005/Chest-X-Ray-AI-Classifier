import os
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models  # type: ignore[import-not-found]
    from tensorflow.keras.applications import MobileNetV2  # type: ignore[import-not-found]
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run this project. Install it with `pip install tensorflow`."
    ) from exc


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset location:
# Set CHEST_XRAY_DATASET in your environment if the dataset
# is stored outside the project directory.
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
print("MobileNetV2 Transfer Learning")
print("=" * 60)

print(f"Project directory : {PROJECT_DIR}")
print(f"Dataset directory : {DATASET_DIR}")
print(f"Model directory   : {MODEL_DIR}")
print(f"Results directory : {RESULTS_DIR}")


# ============================================================
# LOAD DATASET
# ============================================================

train_dir = DATASET_DIR / "train"

if not train_dir.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{train_dir}"
    )

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
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names

print("\nClass names:")
print(class_names)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
    ],
    name="data_augmentation"
)


# ============================================================
# MOBILE NET V2 BASE MODEL
# ============================================================

base_model = MobileNetV2(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained model initially
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=IMAGE_SIZE + (3,),
    name="input_image"
)

x = data_augmentation(inputs)

x = layers.Lambda(
    preprocess_input,
    name="mobilenetv2_preprocess"
)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="prediction"
)(x)

model = models.Model(
    inputs,
    outputs,
    name="MobileNetV2_Transfer"
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
        tf.keras.metrics.Recall(name="recall"),
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint_path = MODEL_DIR / "mobilenetv2_transfer.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = MODEL_DIR / "mobilenetv2_transfer.keras"

model.save(final_model_path)

print("\nModel saved successfully:")
print(final_model_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

import matplotlib.pyplot as plt

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
plt.title("MobileNetV2 Transfer Learning Accuracy")
plt.legend()
plt.tight_layout()

accuracy_path = RESULTS_DIR / "mobilenetv2_accuracy.png"

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
plt.title("MobileNetV2 Transfer Learning Loss")
plt.legend()
plt.tight_layout()

loss_path = RESULTS_DIR / "mobilenetv2_loss.png"

plt.savefig(loss_path)
plt.close()


print("\nTraining graphs saved:")
print(accuracy_path)
print(loss_path)

print("\nTransfer learning completed successfully.")