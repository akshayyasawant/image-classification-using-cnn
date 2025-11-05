import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pandas as pd
import gdown
import os

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Image Classification App", layout="wide")

st.title("🌍 Scene Image Classification using CNN")
st.write("Upload images to classify them into one of the natural scene categories.")

# --------------------------
# CLASS LABELS
# --------------------------
CLASS_LABELS = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --------------------------
# DOWNLOAD MODEL IF NOT PRESENT
# --------------------------
MODEL_PATH = "intel_scene_tuned_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1zhwiS_T8HJz5krT3MlM9-3CT_DBbqOx4"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model... please wait..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def preprocess(img):
    img = img.convert("RGB").resize((128,128))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

def gauge(conf):
    value = conf * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [0,100]}, 'bar': {'color': 'royalblue'}},
        number={'suffix': "%"}
    ))
    fig.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def predict(img):
    x = preprocess(img)
    probs = model.predict(x)[0]
    return CLASS_LABELS[np.argmax(probs)], probs

# --------------------------
# SIDEBAR NAVIGATION
# --------------------------
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Select Mode", ["Single Image Prediction", "Compare Two Images", "Batch Prediction"])

# --------------------------
# MODE 1: SINGLE IMAGE
# --------------------------
if mode == "Single Image Prediction":
    st.header("🖼 Single Image Classification")
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        label, probs = predict(img)
        confidence = np.max(probs)

        st.subheader(f"✅ Predicted Class: **{label.title()}**")
        st.plotly_chart(gauge(confidence), use_container_width=True)

        st.write("### Top-3 Prediction Confidence")
        top3 = np.argsort(probs)[::-1][:3]
        df = pd.DataFrame({
            "Class": [CLASS_LABELS[i].title() for i in top3],
            "Confidence (%)": [round(probs[i]*100,2) for i in top3]
        })
        st.table(df)

# --------------------------
# MODE 2: COMPARE TWO IMAGES
# --------------------------
elif mode == "Compare Two Images":
    st.header("🆚 Compare Two Images")

    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader("Upload First Image", type=["jpg","jpeg","png"], key="img1")
        if file1:
            img1 = Image.open(file1)
            st.image(img1, caption="First Image", use_container_width=True)
            label1, probs1 = predict(img1)

    with col2:
        file2 = st.file_uploader("Upload Second Image", type=["jpg","jpeg","png"], key="img2")
        if file2:
            img2 = Image.open(file2)
            st.image(img2, caption="Second Image", use_container_width=True)
            label2, probs2 = predict(img2)

    if file1 and file2:
        st.subheader("Comparison Result")
        result_df = pd.DataFrame({
            "Image": ["Image 1", "Image 2"],
            "Predicted Class": [label1.title(), label2.title()],
            "Confidence (%)": [round(np.max(probs1)*100,2), round(np.max(probs2)*100,2)]
        })
        st.table(result_df)

# --------------------------
# MODE 3: BATCH PREDICTION
# --------------------------
elif mode == "Batch Prediction":
    st.header("📦 Batch Image Prediction")
    files = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if files:
        result = []
        for file in files:
            img = Image.open(file)
            label, probs = predict(img)
            result.append([file.name, label.title(), round(np.max(probs)*100,2)])

        result_df = pd.DataFrame(result, columns=["File Name", "Predicted Class", "Confidence (%)"])
        st.table(result_df)
