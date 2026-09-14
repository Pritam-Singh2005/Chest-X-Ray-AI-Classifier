from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score
)

# =========================
# PATHS
# =========================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = Path(
    r"D:\chest_xray_classifier\models\mobilenetv2_transfer.keras"
)

RESULTS_DIR = Path(
    r"D:\chest_xray_classifier\results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# SETTINGS
# =========================

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16

# =========================
# LOAD MODEL
# =========================

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# =========================
# LOAD TEST DATA
# =========================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

print("\nClasses:")
print(class_names)

# =========================
# PREDICTIONS
# =========================

print("\nGenerating predictions...")

y_true = []
y_prob = []

for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(labels.numpy().flatten())
    y_prob.extend(predictions.flatten())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)

# =========================
# METRICS
# =========================

accuracy = np.mean(y_true == y_pred)

precision = np.sum(
    (y_pred == 1) & (y_true == 1)
) / max(np.sum(y_pred == 1), 1)

recall = np.sum(
    (y_pred == 1) & (y_true == 1)
) / max(np.sum(y_true == 1), 1)

f1 = f1_score(y_true, y_pred)

roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

specificity = tn / max((tn + fp), 1)

# =========================
# PRINT RESULTS
# =========================

print("\n" + "=" * 60)
print("MOBILENETV2 TEST RESULTS")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")
print(f"Specificity : {specificity:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# =========================
# CONFUSION MATRIX PLOT
# =========================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("MobileNetV2 Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    class_names
)

plt.yticks(
    [0, 1],
    class_names
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = RESULTS_DIR / "mobilenetv2_confusion_matrix.png"

plt.savefig(cm_path)

plt.close()

print(f"\nConfusion matrix saved to:")
print(cm_path)

# =========================
# SAVE RESULTS
# =========================

results_path = RESULTS_DIR / "mobilenetv2_test_results.txt"

with open(results_path, "w") as f:

    f.write("MOBILENETV2 TEST RESULTS\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report:\n")

    f.write(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

print(f"\nResults saved to:")
print(results_path)

print("\nStep 13 completed.")