import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Chest X-Ray AI Classifier",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = Path("models/mobilenetv2_finetuned.keras")
IMAGE_SIZE = (160, 160)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        font-size: 22px;
        font-weight: 600;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    .footer {
        text-align: center;
        color: #777;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🩻 Project")

    st.write("### Chest X-Ray AI Classifier")

    st.write(
        "A deep learning application for binary "
        "classification of chest X-ray images."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("**Architecture:** MobileNetV2")

    st.write("**Learning:** Transfer Learning")

    st.write("**Fine-Tuning:** Enabled")

    st.write("**Input:** 160 × 160 × 3")

    st.write("**Classes:**")

    st.write("• NORMAL")

    st.write("• PNEUMONIA")

    st.divider()

    st.subheader("📊 Test Performance")

    st.metric("Accuracy", "88.30%")

    st.metric("Recall", "95.13%")

    st.metric("ROC-AUC", "95.83%")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write("**Pritam Singh**")

    st.write("B.Tech CSE AI-ML")

    st.markdown(
        "🔗 [GitHub](https://github.com/Pritam-Singh2005)"
    )

    st.markdown(
        "🔗 [LinkedIn](https://www.linkedin.com/in/pritam-singh-920943341/)"
    )

    st.divider()

    st.caption(
        "⚠️ Educational / Research Project"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩻 Chest X-Ray AI Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning + Transfer Learning + Streamlit'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "⚠️ This application is designed for educational and "
    "research purposes only. It is not a medical diagnostic "
    "system and should not be used for clinical decisions."
)


# =========================================================
# MODEL CHECK
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        f"❌ Model file not found.\n\n"
        f"Expected location:\n`{MODEL_PATH}`"
    )

    st.stop()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


try:

    model = load_model()

except Exception as e:

    st.error(f"❌ Failed to load model: {e}")

    st.stop()


# =========================================================
# MODEL STATUS
# =========================================================

st.success(
    "✅ Fine-tuned MobileNetV2 model loaded successfully."
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Prediction",
        "🤖 Model Information",
        "📊 Performance"
    ]
)


# =========================================================
# TAB 1 - PREDICTION
# =========================================================

with tab1:

    st.subheader("📤 Upload Chest X-Ray")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        with col1:

            st.subheader("🖼️ X-Ray Image")

            st.image(
                image,
                caption="Uploaded Chest X-Ray",
                width="stretch"
            )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with col2:

            st.subheader("🔬 AI Prediction")

            if st.button(
                "🔍 Analyze X-Ray",
                type="primary",
                width="stretch"
            ):

                with st.spinner(
                    "Analyzing X-ray..."
                ):

                    # Resize
                    resized_image = image.resize(
                        IMAGE_SIZE
                    )

                    # NumPy conversion
                    image_array = np.array(
                        resized_image
                    )

                    # Batch dimension
                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # MobileNetV2 preprocessing
                    image_array = (
                        tf.keras.applications
                        .mobilenet_v2
                        .preprocess_input(
                            image_array
                        )
                    )

                    # Prediction
                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )

                    probability = float(
                        prediction[0][0]
                    )

                    # Classification
                    if probability >= 0.5:

                        result = "PNEUMONIA"

                        confidence = (
                            probability * 100
                        )

                        normal_probability = (
                            (1 - probability) * 100
                        )

                    else:

                        result = "NORMAL"

                        confidence = (
                            (1 - probability) * 100
                        )

                        normal_probability = (
                            (1 - probability) * 100
                        )


                # -------------------------------------------------
                # RESULT
                # -------------------------------------------------

                st.divider()

                if result == "PNEUMONIA":

                    st.error(
                        "🫁 Prediction: PNEUMONIA"
                    )

                else:

                    st.success(
                        "✅ Prediction: NORMAL"
                    )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                # -------------------------------------------------
                # PROBABILITY
                # -------------------------------------------------

                st.write(
                    "**Prediction Probability**"
                )

                pneumonia_probability = (
                    probability * 100
                )

                st.progress(
                    min(
                        max(
                            pneumonia_probability / 100,
                            0.0
                        ),
                        1.0
                    )
                )

                st.write(
                    f"Pneumonia probability: "
                    f"**{pneumonia_probability:.2f}%**"
                )

                st.write(
                    f"Normal probability: "
                    f"**{100 - pneumonia_probability:.2f}%**"
                )

                # -------------------------------------------------
                # TECHNICAL DETAILS
                # -------------------------------------------------

                with st.expander(
                    "🔬 Technical Prediction Details"
                ):

                    st.write(
                        f"**Raw model output:** "
                        f"{probability:.6f}"
                    )

                    st.write(
                        "**Decision threshold:** 0.50"
                    )

                    st.write(
                        "**Image size:** 160 × 160"
                    )

                    st.write(
                        "**Color channels:** RGB"
                    )

                    st.write(
                        "**Preprocessing:** "
                        "MobileNetV2 preprocess_input"
                    )


# =========================================================
# TAB 2 - MODEL INFORMATION
# =========================================================

with tab2:

    st.subheader("🤖 Model Architecture")

    st.write(
        "The final model uses MobileNetV2 with transfer "
        "learning from ImageNet followed by fine-tuning "
        "of the last layers."
    )

    st.markdown(
        """
        ### Architecture

        ```text
        Input Image
        160 × 160 × 3
              ↓
        MobileNetV2
        ImageNet Pretrained
              ↓
        Global Average Pooling
              ↓
        Dropout (0.3)
              ↓
        Dense (1)
              ↓
        Sigmoid Activation
              ↓
        NORMAL / PNEUMONIA
        ```
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("⚙️ Training Configuration")

        st.write("**Optimizer:** Adam")

        st.write("**Learning Rate:** 0.00001")

        st.write("**Loss:** Binary Crossentropy")

        st.write("**Batch Size:** 16")

        st.write("**Input Size:** 160 × 160")

    with col2:

        st.subheader("🧠 Transfer Learning")

        st.write(
            "MobileNetV2 was initialized with "
            "ImageNet pretrained weights."
        )

        st.write(
            "Most layers were frozen initially."
        )

        st.write(
            "The final 30 layers were fine-tuned."
        )

        st.write(
            "Batch Normalization layers remained frozen."
        )


# =========================================================
# TAB 3 - PERFORMANCE
# =========================================================

with tab3:

    st.subheader(
        "📊 Final Model Performance"
    )

    st.write(
        "Performance measured on the untouched test dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            "88.30%"
        )

    with col2:

        st.metric(
            "Precision",
            "87.29%"
        )

    with col3:

        st.metric(
            "Recall",
            "95.13%"
        )

    with col4:

        st.metric(
            "ROC-AUC",
            "95.83%"
        )

    st.divider()

    st.subheader(
        "📋 Confusion Matrix"
    )

    st.write(
        "The final test results produced the following "
        "confusion matrix:"
    )

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:

        st.write("**NORMAL = 0**")

        st.write("**PNEUMONIA = 1**")

        st.markdown(
            """
            | | Predicted NORMAL | Predicted PNEUMONIA |
            |---|---:|---:|
            | Actual NORMAL | 180 | 54 |
            | Actual PNEUMONIA | 19 | 371 |
            """
        )

    with cm_col2:

        st.metric(
            "Specificity",
            "76.92%"
        )

        st.metric(
            "F1 Score",
            "91.04%"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    🩻 <b>Chest X-Ray Pneumonia Classifier</b><br>

    Built using TensorFlow • MobileNetV2 • Transfer Learning • Streamlit

    <br><br>

    ⚠️ For educational and research purposes only.
    This application does not provide medical diagnosis.

    </div>
    """,
    unsafe_allow_html=True
)