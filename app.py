# Filename: app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# 1️⃣ App title and description
# -----------------------------
st.set_page_config(page_title="Drowsiness Detection AI", layout="centered")
st.title("Drowsiness Detection via Eye State Classification")
st.markdown("""
This application predicts whether a person's eyes are **Open or Closed** using a CNN/Transfer Learning model.
It also provides a **professional evaluation** with confusion matrix visualization.
""")

# -----------------------------
# 2️⃣ Load trained model
# -----------------------------
@st.cache_resource
def load_trained_model():
    model_path = "drowsiness_mobilenet_final.h5"  # your trained model
    model = load_model(model_path)
    return model

model = load_trained_model()
IMG_WIDTH, IMG_HEIGHT = 128, 128

# -----------------------------
# 3️⃣ Image Upload & Prediction
# -----------------------------
st.header("Predict Eye State from Image")

uploaded_file = st.file_uploader("Upload an image of eyes", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    image = load_img(uploaded_file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict
    pred_prob = model.predict(image)[0][0]
    if pred_prob >= 0.5:
        label = "Closed"
    else:
        label = "Open"
    
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** {pred_prob*100:.2f}%")

# -----------------------------
# 4️⃣ Confusion Matrix Heatmap (Validation)
# -----------------------------
st.header("Validation Confusion Matrix Heatmap")

# Dummy example values (replace with your actual validation y_true/y_pred)
y_true = np.array([...])  # e.g., validation labels
y_pred = np.array([...])  # e.g., predicted labels from validation set

if len(y_true) > 0:
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Open','Closed']
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix Heatmap")
    st.pyplot(fig)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    st.header("Classification Report")
    st.table({
        "Class": classes,
        "Precision": [report[c]['precision'] for c in classes],
        "Recall": [report[c]['recall'] for c in classes],
        "F1-score": [report[c]['f1-score'] for c in classes],
        "Support": [report[c]['support'] for c in classes]
    })

# -----------------------------
# 5️⃣ Footer / About
# -----------------------------
st.markdown("---")
st.markdown("**Developed by Hiba Dadda – Master ISI | Drowsiness Detection in Automotive Applications**")
