# ============================================================
# STEP 17: FINAL COMPARISON OF ALL 4 MODELS
# ============================================================

from pathlib import Path
import numpy as np
import tensorflow as tf
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

PROJECT_DIR = Path(r"D:\chest_xray_classifier")

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "Baseline CNN": MODEL_DIR / "baseline_cnn.keras",

    "Weighted CNN": MODEL_DIR / "weighted_cnn.keras",

    "MobileNetV2 Transfer": MODEL_DIR / "mobilenetv2_transfer.keras",

    "MobileNetV2 Fine-Tuned": MODEL_DIR / "mobilenetv2_finetuned.keras",
}


# ============================================================
# 3. INPUT SIZE FOR EACH MODEL
# ============================================================

# Baseline CNN and Weighted CNN were trained at 224x224.
# MobileNetV2 models were trained at 160x160.

IMAGE_SIZES = {
    "Baseline CNN": (224, 224),
    "Weighted CNN": (224, 224),
    "MobileNetV2 Transfer": (160, 160),
    "MobileNetV2 Fine-Tuned": (160, 160),
}


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

def load_test_dataset(image_size):

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=image_size,
        batch_size=16,
        shuffle=False
    )

    return test_dataset


# ============================================================
# 5. CONVERT DATASET TO NUMPY
# ============================================================

def dataset_to_numpy(dataset):

    images = []
    labels = []

    for batch_images, batch_labels in dataset:

        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()

    return images, labels


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

def evaluate_model(model_name, model_path):

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    image_size = IMAGE_SIZES[model_name]

    print(f"Model path : {model_path}")
    print(f"Image size : {image_size}")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not model_path.exists():

        print(f"ERROR: Model not found:")
        print(model_path)

        return None

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    test_dataset = load_test_dataset(image_size)

    print("Loading test images...")

    X_test, y_test = dataset_to_numpy(test_dataset)

    print(f"Test images: {len(X_test)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model = tf.keras.models.load_model(model_path)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("Generating predictions...")

    y_prob = model.predict(
        X_test,
        batch_size=16,
        verbose=1
    ).flatten()

    # Convert probability to class
    y_pred = (y_prob >= 0.5).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults")
    print("-" * 40)

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
            y_test,
            y_pred,
            target_names=["NORMAL", "PNEUMONIA"],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "confusion_matrix": cm
    }


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CHEST X-RAY CLASSIFICATION")
    print("FINAL COMPARISON OF ALL 4 MODELS")
    print("=" * 70)

    print("\nTest dataset:")
    print(TEST_DIR)

    print("\nIMPORTANT:")
    print("The original test dataset is used for final evaluation.")
    print("No training is performed in this script.")

    # --------------------------------------------------------
    # Evaluate all models
    # --------------------------------------------------------

    results = {}

    for model_name, model_path in MODEL_PATHS.items():

        result = evaluate_model(
            model_name,
            model_path
        )

        if result is not None:

            results[model_name] = result


    # ========================================================
    # 8. CHECK RESULTS
    # ========================================================

    if len(results) == 0:

        print("\nNo models were successfully evaluated.")

        return


    # ========================================================
    # 9. PRINT FINAL COMPARISON TABLE
    # ========================================================

    print("\n")
    print("=" * 110)
    print("FINAL MODEL COMPARISON")
    print("=" * 110)

    header = (
        f"{'Model':<28}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'ROC-AUC':<12}"
        f"{'Specificity':<12}"
    )

    print(header)
    print("-" * 110)

    for model_name, result in results.items():

        print(
            f"{model_name:<28}"
            f"{result['accuracy']:<12.4f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['roc_auc']:<12.4f}"
            f"{result['specificity']:<12.4f}"
        )


    # ========================================================
    # 10. FIND BEST MODEL FOR EACH METRIC
    # ========================================================

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "specificity"
    ]

    print("\n")
    print("=" * 70)
    print("BEST MODEL BY METRIC")
    print("=" * 70)

    best_models = {}

    for metric in metrics:

        best_model = max(
            results,
            key=lambda model: results[model][metric]
        )

        best_value = results[best_model][metric]

        best_models[metric] = best_model

        print(
            f"{metric.upper():<15}: "
            f"{best_model} "
            f"({best_value:.4f})"
        )


    # ========================================================
    # 11. CREATE COMPARISON TEXT FILE
    # ========================================================

    text_file = RESULTS_DIR / "all_models_comparison.txt"

    with open(text_file, "w", encoding="utf-8") as f:

        f.write("=" * 90 + "\n")
        f.write("CHEST X-RAY CLASSIFICATION\n")
        f.write("FINAL COMPARISON OF ALL 4 MODELS\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Test Dataset: {TEST_DIR}\n\n")

        f.write("Models evaluated:\n")
        f.write("1. Baseline CNN\n")
        f.write("2. Weighted CNN\n")
        f.write("3. MobileNetV2 Transfer Learning\n")
        f.write("4. MobileNetV2 Fine-Tuned\n\n")

        f.write("=" * 90 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 90 + "\n\n")

        for model_name, result in results.items():

            f.write(f"{model_name}\n")
            f.write("-" * 50 + "\n")

            f.write(
                f"Accuracy    : {result['accuracy']:.4f}\n"
            )

            f.write(
                f"Precision   : {result['precision']:.4f}\n"
            )

            f.write(
                f"Recall      : {result['recall']:.4f}\n"
            )

            f.write(
                f"F1 Score    : {result['f1']:.4f}\n"
            )

            f.write(
                f"ROC-AUC     : {result['roc_auc']:.4f}\n"
            )

            f.write(
                f"Specificity : {result['specificity']:.4f}\n"
            )

            f.write("\nConfusion Matrix:\n")

            f.write(
                str(result["confusion_matrix"])
            )

            f.write("\n\n")


        f.write("=" * 90 + "\n")
        f.write("BEST MODEL BY METRIC\n")
        f.write("=" * 90 + "\n\n")

        for metric in metrics:

            best_model = best_models[metric]
            best_value = results[best_model][metric]

            f.write(
                f"{metric.upper():<15}: "
                f"{best_model} "
                f"({best_value:.4f})\n"
            )


    print("\nComparison results saved to:")
    print(text_file)


    # ========================================================
    # 12. CREATE ACCURACY GRAPH
    # ========================================================

    model_names = list(results.keys())

    accuracies = [
        results[name]["accuracy"]
        for name in model_names
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        model_names,
        accuracies
    )

    plt.title(
        "Accuracy Comparison of All Models"
    )

    plt.ylabel("Accuracy")

    plt.ylim(0, 1)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    for i, value in enumerate(accuracies):

        plt.text(
            i,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    plt.tight_layout()

    accuracy_graph = (
        RESULTS_DIR /
        "all_models_accuracy.png"
    )

    plt.savefig(
        accuracy_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nAccuracy graph saved to:")
    print(accuracy_graph)


    # ========================================================
    # 13. CREATE ALL-METRICS GRAPH
    # ========================================================

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "specificity"
    ]

    x = np.arange(len(model_names))

    width = 0.12

    plt.figure(figsize=(14, 7))

    for i, metric in enumerate(metric_names):

        values = [
            results[name][metric]
            for name in model_names
        ]

        plt.bar(
            x + (i - 2.5) * width,
            values,
            width,
            label=metric.upper()
        )

    plt.title(
        "Performance Comparison of All Models"
    )

    plt.ylabel("Score")

    plt.ylim(0, 1.1)

    plt.xticks(
        x,
        model_names,
        rotation=20,
        ha="right"
    )

    plt.legend()

    plt.tight_layout()

    metrics_graph = (
        RESULTS_DIR /
        "all_models_metrics.png"
    )

    plt.savefig(
        metrics_graph,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nMetrics graph saved to:")
    print(metrics_graph)


    # ========================================================
    # 14. CREATE CONFUSION MATRIX GRAPHS
    # ========================================================

    for model_name, result in results.items():

        cm = result["confusion_matrix"]

        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        plt.figure(figsize=(6, 5))

        plt.imshow(cm)

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.xticks(
            [0, 1],
            ["NORMAL", "PNEUMONIA"]
        )

        plt.yticks(
            [0, 1],
            ["NORMAL", "PNEUMONIA"]
        )

        # Add values inside cells

        for i in range(2):

            for j in range(2):

                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14
                )

        plt.tight_layout()

        cm_path = (
            RESULTS_DIR /
            f"{safe_name}_confusion_matrix.png"
        )

        plt.savefig(
            cm_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Confusion matrix saved: {cm_path}"
        )


    # ========================================================
    # 15. FINAL RECOMMENDATION
    # ========================================================

    # We use F1 score as the primary overall metric because
    # it balances precision and recall.

    best_overall_model = max(
        results,
        key=lambda model: results[model]["f1"]
    )

    best_f1 = results[
        best_overall_model
    ]["f1"]

    print("\n")
    print("=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    print(
        f"\nBest overall model based on F1 Score:"
    )

    print(
        f"{best_overall_model}"
    )

    print(
        f"F1 Score: {best_f1:.4f}"
    )

    print("\nThis model provides the best balance")
    print("between precision and recall among the")
    print("models evaluated on the same test dataset.")

    print("\n")
    print("=" * 70)
    print("STEP 17 COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()