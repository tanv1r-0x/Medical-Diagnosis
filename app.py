"""
Streamlit interface for the Chest X-Ray Diagnosis System (educational).
Run with: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from PIL import Image

from predict import predict

st.set_page_config(page_title="Chest X-Ray Diagnosis", page_icon="🩻", layout="centered")
st.title("🩻 Chest X-Ray Diagnosis System")
st.caption("CNN (Transfer Learning - MobileNetV2) trained to classify Normal vs Pneumonia")

st.warning(
    "Educational project only — not a certified medical device. "
    "Never use this to make real diagnosis or treatment decisions. "
    "Always consult a qualified radiologist/doctor."
)

uploaded = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(tmp.name)
    tmp.close()

    if st.button("Predict", type="primary"):
        try:
            result = predict(tmp.name)
        except FileNotFoundError as e:
            st.error(str(e))
        else:
            st.subheader(f"Prediction: {result['prediction']}")
            st.write(f"Confidence: {result['confidence']:.2%}")
            st.progress(result["confidence"])

    os.remove(tmp.name)
