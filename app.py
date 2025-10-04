import streamlit as st
from braintumor import predict_tumor
import os

st.set_page_config(page_title="🧠 Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor MRI Detection")
st.write("Upload an MRI image and the model will predict if it has a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded MRI Image", use_column_width=True)

    st.write("🔍 Analyzing...")
    try:
        predicted_class, confidence = predict_tumor(img_path)
        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
