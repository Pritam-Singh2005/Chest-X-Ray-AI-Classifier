# 🩻 Chest X-Ray Pneumonia Classifier

A deep learning based chest X-ray image classification system that uses **MobileNetV2 Transfer Learning and Fine-Tuning** to classify X-ray images into two categories:

- NORMAL
- PNEUMONIA

The project includes model training, evaluation, comparison, image prediction and an interactive Streamlit web application.

---

## 🚀 Demo

The application provides a simple web interface where users can upload a chest X-ray image and receive a model prediction.

### Main Features

- 🩻 Chest X-ray image upload
- 🤖 MobileNetV2 transfer learning
- 🔧 Fine-tuning
- 📊 Model evaluation
- 🔍 Single-image prediction
- 📈 Confidence/probability display
- 🌐 Streamlit web application
- 📋 Model performance dashboard

---

## 🧠 Project Workflow

```text
Chest X-Ray Dataset
        ↓
Data Preprocessing
        ↓
Train / Validation Split
        ↓
Baseline CNN
        ↓
Class-Weighted CNN
        ↓
MobileNetV2 Transfer Learning
        ↓
MobileNetV2 Fine-Tuning
        ↓
Model Evaluation
        ↓
Final Model Selection
        ↓
Streamlit Application
        ↓
NORMAL / PNEUMONIA