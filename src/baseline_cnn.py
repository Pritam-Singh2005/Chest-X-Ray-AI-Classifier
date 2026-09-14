import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


# ==========================================
# 1. DATASET PATH
# ==========================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


# ==========================================
# 2. PARAMETERS
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
EPOCHS = 10


# ==========================================
# 3. LOAD TRAINING DATA
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# 5. LOAD TEST DATA
# ==========================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 6. PRINT CLASSES
# ==========================================

print("\nClasses:")
print(train_dataset.class_names)


# ==========================================
# 7. DATA PREFETCHING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ==========================================
# 8. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.05),
], name="data_augmentation")


# ==========================================
# 9. BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    # Input
    tf.keras.layers.Input(
        shape=(224, 224, 3)
    ),

    # Data augmentation
    data_augmentation,

    # Normalize pixels
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # --------------------------------------
    # Convolution Block 1
    # --------------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------
    # Convolution Block 2
    # --------------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------
    # Convolution Block 3
    # --------------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------
    # Classification section
    # --------------------------------------

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    # Binary classification
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# 10. DISPLAY MODEL
# ==========================================

print("\n========== MODEL SUMMARY ==========\n")

model.summary()


# ==========================================
# 11. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
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


# ==========================================
# 12. TRAIN MODEL
# ==========================================

print("\n========== STARTING TRAINING ==========\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    exist_ok=True
)

model_path = MODEL_DIR / "baseline_cnn.keras"

model.save(model_path)

print("\nModel saved to:")
print(model_path)


# ==========================================
# 14. TEST MODEL
# ==========================================

print("\n========== TEST RESULTS ==========\n")

test_results = model.evaluate(
    test_dataset,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        f"{name}: {value:.4f}"
    )


# ==========================================
# 15. PLOT TRAINING ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training vs Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================
# 16. PLOT LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training vs Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.show()


print("\n========== TRAINING COMPLETE ==========")