import tensorflow as tf
from pathlib import Path


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
# 6. PRINT CLASS NAMES
# ==========================================

print("\n========================================")
print("           CLASS INFORMATION")
print("========================================")

print("Classes:", train_dataset.class_names)


# ==========================================
# 7. PRINT DATASET INFORMATION
# ==========================================

print("\n========================================")
print("          DATASET INFORMATION")
print("========================================")

print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nTraining batches:",
      tf.data.experimental.cardinality(train_dataset).numpy())

print("Validation batches:",
      tf.data.experimental.cardinality(validation_dataset).numpy())

print("Test batches:",
      tf.data.experimental.cardinality(test_dataset).numpy())


# ==========================================
# 8. PREFETCH DATA
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 9. CHECK ONE BATCH
# ==========================================

for images, labels in train_dataset.take(1):

    print("\n========================================")
    print("            BATCH CHECK")
    print("========================================")

    print("Image tensor shape:", images.shape)
    print("Label tensor shape:", labels.shape)

    print("Image data type:", images.dtype)
    print("Label data type:", labels.dtype)

    print("\nFirst 5 labels:")
    print(labels[:5].numpy().flatten())

    break


print("\n========================================")
print("       DATA LOADING SUCCESSFUL")
print("========================================")