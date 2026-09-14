import tensorflow as tf
from pathlib import Path
import numpy as np


# ==========================================
# 1. PATHS
# ==========================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path(
    r"D:\chest_xray_classifier\models\baseline_cnn.keras"
)


# ==========================================
# 2. PARAMETERS
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ==========================================
# 3. CHECK MODEL
# ==========================================

print("\n" + "=" * 55)
print("       BASELINE CNN MODEL EVALUATION")
print("=" * 55)

print("\nModel path:")
print(MODEL_PATH)

if not MODEL_PATH.exists():

    print("\n❌ Model not found!")
    print("Check that baseline_cnn.keras exists inside models.")
    exit()

print("\n✅ Model found!")


# ==========================================
# 4. LOAD MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded successfully!")


# ==========================================
# 5. LOAD TEST DATA
# ==========================================

print("\nLoading test dataset...")

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

print("\nClasses:")

print(test_dataset.class_names)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

print("\n" + "=" * 55)
print("              TEST EVALUATION")
print("=" * 55)

results = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nRaw results:")
print(results)


# ==========================================
# 8. DISPLAY METRICS
# ==========================================

print("\n" + "=" * 55)
print("              MODEL METRICS")
print("=" * 55)

for metric_name, value in zip(
    model.metrics_names,
    results
):

    print(
        f"{metric_name}: {value:.4f}"
    )


# ==========================================
# 9. MAKE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_dataset,
    verbose=1
)


# ==========================================
# 10. CONVERT PROBABILITIES TO LABELS
# ==========================================

predicted_labels = (
    predictions.ravel() >= 0.5
).astype(int)


# ==========================================
# 11. GET TRUE LABELS
# ==========================================

true_labels = np.concatenate(
    [
        labels.numpy().ravel()
        for images, labels in test_dataset
    ]
).astype(int)


# ==========================================
# 12. CHECK NUMBER OF PREDICTIONS
# ==========================================

print("\n" + "=" * 55)
print("            PREDICTION CHECK")
print("=" * 55)

print("\nNumber of test images:",
      len(true_labels))

print("Number of predictions:",
      len(predicted_labels))


# ==========================================
# 13. PREDICTION DISTRIBUTION
# ==========================================

normal_predictions = np.sum(
    predicted_labels == 0
)

pneumonia_predictions = np.sum(
    predicted_labels == 1
)

print("\nPredicted NORMAL:",
      normal_predictions)

print("Predicted PNEUMONIA:",
      pneumonia_predictions)


# ==========================================
# 14. TRUE LABEL DISTRIBUTION
# ==========================================

normal_actual = np.sum(
    true_labels == 0
)

pneumonia_actual = np.sum(
    true_labels == 1
)

print("\nActual NORMAL:",
      normal_actual)

print("Actual PNEUMONIA:",
      pneumonia_actual)


# ==========================================
# 15. CONFUSION MATRIX VALUES
# ==========================================

true_negative = np.sum(
    (true_labels == 0) &
    (predicted_labels == 0)
)

false_positive = np.sum(
    (true_labels == 0) &
    (predicted_labels == 1)
)

false_negative = np.sum(
    (true_labels == 1) &
    (predicted_labels == 0)
)

true_positive = np.sum(
    (true_labels == 1) &
    (predicted_labels == 1)
)


# ==========================================
# 16. PRINT CONFUSION MATRIX
# ==========================================

print("\n" + "=" * 55)
print("            CONFUSION MATRIX")
print("=" * 55)

print()

print("                 Predicted")
print("              NORMAL  PNEUMONIA")

print(
    f"Actual NORMAL   {true_negative:4d}      {false_positive:4d}"
)

print(
    f"Actual PNEUMONIA{false_negative:4d}      {true_positive:4d}"
)


# ==========================================
# 17. MANUAL METRICS
# ==========================================

total = (
    true_negative +
    false_positive +
    false_negative +
    true_positive
)

accuracy = (
    true_positive + true_negative
) / total

precision = (
    true_positive /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1_score = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ==========================================
# 18. PRINT FINAL METRICS
# ==========================================

print("\n" + "=" * 55)
print("          CALCULATED METRICS")
print("=" * 55)

print(
    f"\nAccuracy:  {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall:    {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1-score:  {f1_score:.4f} "
    f"({f1_score * 100:.2f}%)"
)


# ==========================================
# 19. FINISH
# ==========================================

print("\n" + "=" * 55)
print("          EVALUATION COMPLETE")
print("=" * 55)