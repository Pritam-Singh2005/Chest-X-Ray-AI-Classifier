from pathlib import Path

import numpy as np
import tensorflow as tf

from PIL import Image

# pyright: reportMissingImports=false
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR /
    "models" /
    "mobilenetv2_finetuned.keras"
)

IMAGE_SIZE = (160, 160)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}\n\n"
        "Make sure the fine-tuned model exists "
        "inside the models folder."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_xray(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Resize
    image = image.resize(
        IMAGE_SIZE
    )

    # Convert to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Prediction
    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # Classification threshold
    threshold = 0.5

    if probability >= threshold:

        prediction = "PNEUMONIA"
        confidence = probability

    else:

        prediction = "NORMAL"
        confidence = 1 - probability


    return prediction, confidence, probability


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Chest X-Ray Pneumonia Classifier")
    print("=" * 60)

    image_path = input(
        "\nEnter the path of the X-ray image: "
    ).strip().strip('"')

    try:

        prediction, confidence, probability = (
            predict_xray(image_path)
        )

        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)

        print(
            f"Prediction : {prediction}"
        )

        print(
            f"Confidence : {confidence * 100:.2f}%"
        )

        print(
            f"Pneumonia probability : "
            f"{probability * 100:.2f}%"
        )

        print("=" * 60)

    except Exception as error:

        print(
            f"\nError: {error}"
        )