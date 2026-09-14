from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PATHS
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = Path(
    r"D:\chest_xray_classifier\models"
)

BASELINE_MODEL = MODEL_DIR / "baseline_cnn.keras"

WEIGHTED_MODEL = MODEL_DIR / "weighted_cnn.keras"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("STEP 11 - BASELINE CNN VS WEIGHTED CNN")
print("=" * 70)


# ============================================================
# 4. CHECK FILES
# ============================================================

print("\nChecking files...")

if not TEST_DIR.exists():
    print("❌ Test directory not found!")
    print(TEST_DIR)
    raise SystemExit

if not BASELINE_MODEL.exists():
    print("❌ Baseline CNN model not found!")
    print(BASELINE_MODEL)
    raise SystemExit

if not WEIGHTED_MODEL.exists():
    print("❌ Weighted CNN model not found!")
    print(WEIGHTED_MODEL)
    raise SystemExit

print("✅ Test dataset found")
print("✅ Baseline CNN found")
print("✅ Weighted CNN found")


# ============================================================
# 5. LOAD TEST DATASET
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

for index, name in enumerate(class_names):
    print(f"{index} = {name}")


# ============================================================
# 6. PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

test_ds = test_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. LOAD MODELS
# ============================================================

print("\nLoading models...")

baseline_model = tf.keras.models.load_model(
    BASELINE_MODEL
)

weighted_model = tf.keras.models.load_model(
    WEIGHTED_MODEL
)

print("✅ Baseline CNN loaded")
print("✅ Weighted CNN loaded")


# ============================================================
# 8. FUNCTION TO GET PREDICTIONS
# ============================================================

def get_predictions(model, dataset):

    y_true = []
    y_probability = []

    for images, labels in dataset:

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

    y_true = np.array(y_true).astype(int)

    y_probability = np.array(
        y_probability
    )

    y_pred = (
        y_probability >= 0.5
    ).astype(int)

    return y_true, y_pred, y_probability


# ============================================================
# 9. EVALUATE BASELINE CNN
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING BASELINE CNN")
print("=" * 70)

baseline_true, baseline_pred, baseline_prob = (
    get_predictions(
        baseline_model,
        test_ds
    )
)


# ============================================================
# 10. BASELINE METRICS
# ============================================================

baseline_accuracy = accuracy_score(
    baseline_true,
    baseline_pred
)

baseline_precision = precision_score(
    baseline_true,
    baseline_pred,
    zero_division=0
)

baseline_recall = recall_score(
    baseline_true,
    baseline_pred,
    zero_division=0
)

baseline_f1 = f1_score(
    baseline_true,
    baseline_pred,
    zero_division=0
)

baseline_auc = roc_auc_score(
    baseline_true,
    baseline_prob
)

baseline_cm = confusion_matrix(
    baseline_true,
    baseline_pred
)


# ============================================================
# 11. EVALUATE WEIGHTED CNN
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING WEIGHTED CNN")
print("=" * 70)

weighted_true, weighted_pred, weighted_prob = (
    get_predictions(
        weighted_model,
        test_ds
    )
)


# ============================================================
# 12. WEIGHTED METRICS
# ============================================================

weighted_accuracy = accuracy_score(
    weighted_true,
    weighted_pred
)

weighted_precision = precision_score(
    weighted_true,
    weighted_pred,
    zero_division=0
)

weighted_recall = recall_score(
    weighted_true,
    weighted_pred,
    zero_division=0
)

weighted_f1 = f1_score(
    weighted_true,
    weighted_pred,
    zero_division=0
)

weighted_auc = roc_auc_score(
    weighted_true,
    weighted_prob
)

weighted_cm = confusion_matrix(
    weighted_true,
    weighted_pred
)


# ============================================================
# 13. SPECIFICITY FUNCTION
# ============================================================

def calculate_specificity(cm):

    if cm.shape != (2, 2):
        return 0.0

    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return 0.0

    return tn / (tn + fp)


baseline_specificity = calculate_specificity(
    baseline_cm
)

weighted_specificity = calculate_specificity(
    weighted_cm
)


# ============================================================
# 14. PRINT COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<20}"
    f"{'Baseline CNN':>20}"
    f"{'Weighted CNN':>20}"
)

print("-" * 70)

print(
    f"{'Accuracy':<20}"
    f"{baseline_accuracy * 100:>19.2f}%"
    f"{weighted_accuracy * 100:>19.2f}%"
)

print(
    f"{'Precision':<20}"
    f"{baseline_precision:>20.4f}"
    f"{weighted_precision:>20.4f}"
)

print(
    f"{'Recall':<20}"
    f"{baseline_recall:>20.4f}"
    f"{weighted_recall:>20.4f}"
)

print(
    f"{'F1 Score':<20}"
    f"{baseline_f1:>20.4f}"
    f"{weighted_f1:>20.4f}"
)

print(
    f"{'ROC-AUC':<20}"
    f"{baseline_auc:>20.4f}"
    f"{weighted_auc:>20.4f}"
)

print(
    f"{'Specificity':<20}"
    f"{baseline_specificity:>20.4f}"
    f"{weighted_specificity:>20.4f}"
)


# ============================================================
# 15. BASELINE CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CNN CONFUSION MATRIX")
print("=" * 70)

print(baseline_cm)

if baseline_cm.shape == (2, 2):

    tn, fp, fn, tp = baseline_cm.ravel()

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)


# ============================================================
# 16. WEIGHTED CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("WEIGHTED CNN CONFUSION MATRIX")
print("=" * 70)

print(weighted_cm)

if weighted_cm.shape == (2, 2):

    tn, fp, fn, tp = weighted_cm.ravel()

    print("\nTN:", tn)
    print("FP:", fp)
    print("FN:", fn)
    print("TP:", tp)


# ============================================================
# 17. BASELINE CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("BASELINE CNN CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        baseline_true,
        baseline_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 18. WEIGHTED CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("WEIGHTED CNN CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        weighted_true,
        weighted_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 19. DETERMINE BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL ANALYSIS")
print("=" * 70)

if weighted_accuracy > baseline_accuracy:

    print(
        "🏆 Weighted CNN has higher test accuracy."
    )

elif baseline_accuracy > weighted_accuracy:

    print(
        "🏆 Baseline CNN has higher test accuracy."
    )

else:

    print(
        "🤝 Both models have the same test accuracy."
    )


if weighted_f1 > baseline_f1:

    print(
        "🏆 Weighted CNN has higher F1-score."
    )

elif baseline_f1 > weighted_f1:

    print(
        "🏆 Baseline CNN has higher F1-score."
    )


if weighted_recall > baseline_recall:

    print(
        "🏆 Weighted CNN has higher recall."
    )

elif baseline_recall > weighted_recall:

    print(
        "🏆 Baseline CNN has higher recall."
    )


# ============================================================
# 20. PLOT ACCURACY COMPARISON
# ============================================================

models = [
    "Baseline CNN",
    "Weighted CNN"
]

accuracies = [
    baseline_accuracy * 100,
    weighted_accuracy * 100
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy (%)")
plt.title("Baseline CNN vs Weighted CNN - Accuracy")

plt.ylim(
    0,
    100
)

for bar, value in zip(
    bars,
    accuracies
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

accuracy_path = (
    MODEL_DIR /
    "baseline_vs_weighted_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.show()


# ============================================================
# 21. PLOT MULTIPLE METRICS
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "Specificity"
]

baseline_values = [
    baseline_accuracy,
    baseline_precision,
    baseline_recall,
    baseline_f1,
    baseline_auc,
    baseline_specificity
]

weighted_values = [
    weighted_accuracy,
    weighted_precision,
    weighted_recall,
    weighted_f1,
    weighted_auc,
    weighted_specificity
]

x = np.arange(
    len(metrics)
)

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline CNN"
)

plt.bar(
    x + width / 2,
    weighted_values,
    width,
    label="Weighted CNN"
)

plt.xticks(
    x,
    metrics,
    rotation=30
)

plt.ylabel("Score")
plt.title(
    "Baseline CNN vs Weighted CNN - Metrics"
)

plt.ylim(
    0,
    1.1
)

plt.legend()

plt.tight_layout()

metrics_path = (
    MODEL_DIR /
    "baseline_vs_weighted_metrics.png"
)

plt.savefig(
    metrics_path,
    dpi=300
)

plt.show()


# ============================================================
# 22. SAVE COMPARISON REPORT
# ============================================================

report_path = (
    MODEL_DIR /
    "model_comparison.txt"
)

with open(
    report_path,
    "w"
) as file:

    file.write(
        "BASELINE CNN VS WEIGHTED CNN\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Baseline Accuracy: "
        f"{baseline_accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Weighted Accuracy: "
        f"{weighted_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        f"Baseline Precision: "
        f"{baseline_precision:.4f}\n"
    )

    file.write(
        f"Weighted Precision: "
        f"{weighted_precision:.4f}\n\n"
    )

    file.write(
        f"Baseline Recall: "
        f"{baseline_recall:.4f}\n"
    )

    file.write(
        f"Weighted Recall: "
        f"{weighted_recall:.4f}\n\n"
    )

    file.write(
        f"Baseline F1: "
        f"{baseline_f1:.4f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{weighted_f1:.4f}\n\n"
    )

    file.write(
        f"Baseline ROC-AUC: "
        f"{baseline_auc:.4f}\n"
    )

    file.write(
        f"Weighted ROC-AUC: "
        f"{weighted_auc:.4f}\n\n"
    )

    file.write(
        f"Baseline Specificity: "
        f"{baseline_specificity:.4f}\n"
    )

    file.write(
        f"Weighted Specificity: "
        f"{weighted_specificity:.4f}\n\n"
    )

    file.write(
        "BASELINE CONFUSION MATRIX\n"
    )

    file.write(
        str(baseline_cm)
    )

    file.write(
        "\n\nWEIGHTED CONFUSION MATRIX\n"
    )

    file.write(
        str(weighted_cm)
    )


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 COMPLETED ✅")
print("=" * 70)

print("\nGenerated files:")

print(
    "1.",
    accuracy_path
)

print(
    "2.",
    metrics_path
)

print(
    "3.",
    report_path
)

print("\nNext step:")
print(
    "Step 12 - MobileNetV2 Transfer Learning"
)