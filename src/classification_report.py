import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score
)


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

print("\n" + "=" * 60)
print("       CNN CLASSIFICATION REPORT")
print("=" * 60)

if not MODEL_PATH.exists():

    print("\n❌ Model not found:")
    print(MODEL_PATH)
    exit()

print("\n✅ Model found")


# ==========================================
# 4. LOAD MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded")


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

class_names = test_dataset.class_names

print("\nClasses:")
print(class_names)


# ==========================================
# 6. GET TRUE LABELS
# ==========================================

print("\nCollecting true labels...")

true_labels = np.concatenate(
    [
        labels.numpy().ravel()
        for images, labels in test_dataset
    ]
).astype(int)


# ==========================================
# 7. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).ravel()


# ==========================================
# 8. CONVERT PROBABILITY TO CLASS
# ==========================================

predicted_labels = (
    probabilities >= 0.5
).astype(int)


# ==========================================
# 9. CHECK LABELS
# ==========================================

print("\n" + "=" * 60)
print("             LABEL INFORMATION")
print("=" * 60)

print("\nClass 0:", class_names[0])
print("Class 1:", class_names[1])

print("\nNumber of test images:", len(true_labels))
print("Number of predictions:", len(predicted_labels))


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\n" + "=" * 60)
print("             CONFUSION MATRIX")
print("=" * 60)

print()

print("                 Predicted")
print(
    f"                 {class_names[0]:<10} {class_names[1]}"
)

print(
    f"Actual {class_names[0]:<10} {cm[0][0]:<10} {cm[0][1]}"
)

print(
    f"Actual {class_names[1]:<10} {cm[1][0]:<10} {cm[1][1]}"
)


# ==========================================
# 11. EXTRACT VALUES
# ==========================================

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]


print("\n" + "=" * 60)
print("          CONFUSION MATRIX VALUES")
print("=" * 60)

print("\nTrue Negative :", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive :", TP)


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("          CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print("\n")
print(report)


# ==========================================
# 13. F1 SCORE
# ==========================================

f1 = f1_score(
    true_labels,
    predicted_labels
)

print("Overall F1-score:")
print(f"{f1:.4f}")


# ==========================================
# 14. ROC-AUC
# ==========================================

try:

    auc = roc_auc_score(
        true_labels,
        probabilities
    )

    print("\nROC-AUC:")
    print(f"{auc:.4f}")

except ValueError:

    print("\nROC-AUC could not be calculated.")


# ==========================================
# 15. VISUAL CONFUSION MATRIX
# ==========================================

print("\nDisplaying confusion matrix...")

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot()

plt.title(
    "Chest X-Ray CNN Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ==========================================
# 16. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("             EVALUATION COMPLETE")
print("=" * 60)

print("\nTN:", TN)
print("FP:", FP)
print("FN:", FN)
print("TP:", TP)

print("\nF1-score:", round(f1, 4))

if "auc" in locals():

    print("ROC-AUC:", round(auc, 4))

print("\n")