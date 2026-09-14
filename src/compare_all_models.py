from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\ps879\Downloads\chest_xray\chest_xray"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = Path(
    r"D:\chest_xray_classifier\models"
)

RESULTS_DIR = Path(
    r"D:\chest_xray_classifier\results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

models = {
    "Baseline CNN": {
        "path": MODEL_DIR / "baseline_cnn.keras",
        "image_size": (224, 224)
    },

    "Weighted CNN": {
        "path": MODEL_DIR / "weighted_cnn.keras",
        "image_size": (224, 224)
    },

    "MobileNetV2": {
        "path": MODEL_DIR / "mobilenetv2_transfer.keras",
        "image_size": (160, 160)
    }
}

BATCH_SIZE = 16

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model_path, image_size):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    print(f"Image size: {image_size}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(model_path)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load test dataset using correct image size
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=image_size,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    class_names = test_ds.class_names

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_true = []
    y_prob = []

    print("\nGenerating predictions...")

    for images, labels in test_ds:

        predictions = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            labels.numpy().flatten()
        )

        y_prob.extend(
            predictions.flatten()
        )

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = np.mean(
        y_true == y_pred
    )

    precision = tp / max(
        tp + fp,
        1
    )

    recall = tp / max(
        tp + fn,
        1
    )

    specificity = tn / max(
        tn + fp,
        1
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "cm": cm
    }


# ============================================================
# EVALUATE MODELS
# ============================================================

results = {}

for model_name, config in models.items():

    model_path = config["path"]
    image_size = config["image_size"]

    if not model_path.exists():

        print(
            f"\nWARNING: {model_name} model not found:"
        )

        print(model_path)

        continue

    results[model_name] = evaluate_model(
        model_name,
        model_path,
        image_size
    )


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n\n")

print("=" * 100)
print("MODEL COMPARISON")
print("=" * 100)

print(
    f"{'Model':<20}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'ROC-AUC':<12}"
    f"{'Specificity':<12}"
)

print("-" * 100)

for name, result in results.items():

    print(
        f"{name:<20}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
        f"{result['roc_auc']:<12.4f}"
        f"{result['specificity']:<12.4f}"
    )

print("=" * 100)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\nCONFUSION MATRICES")
print("=" * 60)

for name, result in results.items():

    print(f"\n{name}:")

    print(
        result["cm"]
    )


# ============================================================
# SAVE TEXT RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "all_models_comparison.txt"
)

with open(results_file, "w") as f:

    f.write(
        "CHEST X-RAY MODEL COMPARISON\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for name, result in results.items():

        f.write(
            f"{name}\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Accuracy    : "
            f"{result['accuracy']:.4f}\n"
        )

        f.write(
            f"Precision   : "
            f"{result['precision']:.4f}\n"
        )

        f.write(
            f"Recall      : "
            f"{result['recall']:.4f}\n"
        )

        f.write(
            f"F1 Score    : "
            f"{result['f1']:.4f}\n"
        )

        f.write(
            f"ROC-AUC     : "
            f"{result['roc_auc']:.4f}\n"
        )

        f.write(
            f"Specificity : "
            f"{result['specificity']:.4f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(result["cm"])
        )

        f.write("\n\n")


print(
    f"\nResults saved to:"
)

print(results_file)


# ============================================================
# ACCURACY GRAPH
# ============================================================

names = list(results.keys())

accuracy_values = [
    results[name]["accuracy"]
    for name in names
]

plt.figure(figsize=(9, 5))

plt.bar(
    names,
    accuracy_values
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

accuracy_graph = (
    RESULTS_DIR /
    "all_models_accuracy.png"
)

plt.savefig(
    accuracy_graph
)

plt.close()


# ============================================================
# COMPLETE METRICS GRAPH
# ============================================================

x = np.arange(
    len(names)
)

width = 0.12

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "specificity"
]

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):

    values = [
        results[name][metric]
        for name in names
    ]

    plt.bar(
        x + (i - 2.5) * width,
        values,
        width,
        label=metric
    )

plt.xticks(
    x,
    names,
    rotation=15
)

plt.ylabel(
    "Score"
)

plt.title(
    "Complete Model Performance Comparison"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

metrics_graph = (
    RESULTS_DIR /
    "all_models_metrics.png"
)

plt.savefig(
    metrics_graph
)

plt.close()


# ============================================================
# BEST MODELS
# ============================================================

best_accuracy = max(
    results,
    key=lambda name:
    results[name]["accuracy"]
)

best_f1 = max(
    results,
    key=lambda name:
    results[name]["f1"]
)

best_recall = max(
    results,
    key=lambda name:
    results[name]["recall"]
)

best_specificity = max(
    results,
    key=lambda name:
    results[name]["specificity"]
)

print("\n" + "=" * 60)
print("BEST MODELS")
print("=" * 60)

print(
    f"Best Accuracy    : "
    f"{best_accuracy}"
)

print(
    f"Best F1 Score    : "
    f"{best_f1}"
)

print(
    f"Best Recall      : "
    f"{best_recall}"
)

print(
    f"Best Specificity : "
    f"{best_specificity}"
)

print("=" * 60)

print("\nSTEP 14 COMPLETED.")