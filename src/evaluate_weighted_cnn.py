from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. PATHS
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path(
    r"D:\chest_xray_classifier\models\weighted_cnn.keras"
)

OUTPUT_DIR = Path(
    r"D:\chest_xray_classifier\models"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 60)
print("STEP 10 - WEIGHTED CNN FINAL TEST EVALUATION")
print("=" * 60)


# ============================================================
# 4. CHECK PATHS
# ============================================================

print("\nChecking files...")

if not TEST_DIR.exists():
    print("❌ Test directory not found!")
    print(TEST_DIR)
    raise SystemExit

if not MODEL_PATH.exists():
    print("❌ Weighted CNN model not found!")
    print(MODEL_PATH)
    print("\nPlease run Step 9 first.")
    raise SystemExit

print("✅ Test dataset found")
print("✅ Weighted CNN model found")


# ============================================================
# 5. LOAD WEIGHTED CNN MODEL
# ============================================================

print("\nLoading Weighted CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")


# ============================================================
# 6. LOAD TEST DATASET
# ============================================================

print("\nLoading untouched test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nClass names:")
print(class_names)

print("\nClass mapping:")

for index, class_name in enumerate(class_names):
    print(f"{index} = {class_name}")


# ============================================================
# 7. PREFETCH DATA
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

test_ds = test_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING WEIGHTED CNN ON TEST DATA")
print("=" * 60)

results = model.evaluate(
    test_ds,
    verbose=1
)


# ============================================================
# 9. DISPLAY KERAS METRICS
# ============================================================

print("\n" + "=" * 60)
print("TEST METRICS")
print("=" * 60)

for metric_name, metric_value in zip(
    model.metrics_names,
    results
):
    print(
        f"{metric_name:12s}: {metric_value:.4f}"
    )


# ============================================================
# 10. GET TRUE LABELS AND PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_probability = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels.numpy().flatten()
    )

    y_probability.extend(
        predictions.flatten()
    )


# Convert to NumPy arrays

y_true = np.array(y_true).astype(int)

y_probability = np.array(y_probability)


# ============================================================
# 11. CONVERT PROBABILITY TO CLASS
# ============================================================

# Sigmoid output:
#
# probability >= 0.5 → class 1
# probability <  0.5 → class 0

y_pred = (
    y_probability >= 0.5
).astype(int)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# 13. EXTRACT TP TN FP FN
# ============================================================

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    print("\nTrue Negative (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)
    print("True Positive (TP):", tp)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================================
# 15. F1 SCORE
# ============================================================

f1 = f1_score(
    y_true,
    y_pred
)

print("=" * 60)
print("F1 SCORE")
print("=" * 60)

print(f"F1 Score: {f1:.4f}")


# ============================================================
# 16. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    y_probability
)

print("\n" + "=" * 60)
print("ROC-AUC")
print("=" * 60)

print(f"ROC-AUC Score: {roc_auc:.4f}")


# ============================================================
# 17. MANUAL ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

print("\n" + "=" * 60)
print("MANUAL TEST ACCURACY")
print("=" * 60)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 18. PRECISION
# ============================================================

if cm.shape == (2, 2):

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    print("\nPrecision:")
    print(f"{precision:.4f}")


# ============================================================
# 19. RECALL / SENSITIVITY
# ============================================================

if cm.shape == (2, 2):

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    print("\nRecall / Sensitivity:")
    print(f"{recall:.4f}")


# ============================================================
# 20. SPECIFICITY
# ============================================================

if cm.shape == (2, 2):

    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    print("\nSpecificity:")
    print(f"{specificity:.4f}")


# ============================================================
# 21. CONFUSION MATRIX VISUALIZATION
# ============================================================

print("\nCreating confusion matrix image...")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title(
    "Weighted CNN - Confusion Matrix"
)

plt.tight_layout()

cm_path = (
    OUTPUT_DIR /
    "weighted_cnn_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.show()

print(
    f"Confusion matrix saved to:\n{cm_path}"
)


# ============================================================
# 22. SAVE TEST RESULTS TO TEXT FILE
# ============================================================

results_path = (
    OUTPUT_DIR /
    "weighted_cnn_test_results.txt"
)

with open(
    results_path,
    "w"
) as file:

    file.write(
        "WEIGHTED CNN TEST RESULTS\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Test Accuracy (%): "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.4f}\n"
    )

    if cm.shape == (2, 2):

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall / Sensitivity: "
            f"{recall:.4f}\n"
        )

        file.write(
            f"Specificity: "
            f"{specificity:.4f}\n"
        )

        file.write(
            f"\nTrue Negative: {tn}\n"
        )

        file.write(
            f"False Positive: {fp}\n"
        )

        file.write(
            f"False Negative: {fn}\n"
        )

        file.write(
            f"True Positive: {tp}\n"
        )

    file.write(
        "\n\nCLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(report)


print(
    f"\nTest results saved to:\n{results_path}"
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 10 COMPLETED ✅")
print("=" * 60)

print("\nWeighted CNN Final Results:")

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"F1 Score    : {f1:.4f}"
)

print(
    f"ROC-AUC     : {roc_auc:.4f}"
)

if cm.shape == (2, 2):

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

print("\nFiles created:")

print(
    "1. weighted_cnn_confusion_matrix.png"
)

print(
    "2. weighted_cnn_test_results.txt"
)

print("\nNext step:")
print(
    "Compare Baseline CNN vs Weighted CNN."
)