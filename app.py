import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import pandas as pd
import gdown
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Neon Scene Classifier", layout="wide")

# -----------------------------------
# CLASS LABELS (Intel Scene Classes)
# -----------------------------------
CLASS_LABELS = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# -----------------------------------
# MODEL DOWNLOAD (Google Drive)
# -----------------------------------
MODEL_PATH = "intel_scene_tuned_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1zhwiS_T8HJz5krT3MlM9-3CT_DBbqOx4"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model (first time only)... Please wait..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# -----------------------------------
# NEON UI STYLING
# -----------------------------------
st.markdown("""
<style>
body, .stApp { background: radial-gradient(circle at top, #0b0f1a, #05070c); color: #eaf4ff; }
.neon-title { text-align:center; font-size:40px; font-weight:800; color:#8afcff; text-shadow:0 0 12px #00dfff; }
.section-header { font-size:22px; font-weight:600; color:#84d9ff; margin-top:15px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------
def preprocess(img):
    img = img.convert("RGB").resize((128,128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def gauge(conf):
    val = conf * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        gauge={'axis': {'range': [0,100]}, 'bar': {'color': '#00eaff'}},
        number={'suffix': "%"}
    ))
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def predict(img):
    x = preprocess(img)
    probs = model.predict(x)[0]
    idx = np.argmax(probs)
    return CLASS_LABELS[idx], probs

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------
st.sidebar.title("📍 Navigation")
mode = st.sidebar.radio("Select Mode:", ["Single Image Prediction", "Compare Two Images", "Batch Prediction"])

st.markdown('<div class="neon-title">🎮 Neon Scene Classifier</div>', unsafe_allow_html=True)

# -----------------------------------
# MODE 1: SINGLE IMAGE
# -----------------------------------
if mode == "Single Image Prediction":
    file = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"])
    
    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        label, probs = predict(img)
        conf = np.max(probs)

        st.markdown(f"### ✅ **Prediction:** {label.title()}")
        st.plotly_chart(gauge(conf), use_container_width=True)

        st.write("### 🎯 Top-3 Predictions:")
        top3 = np.argsort(probs)[::-1][:3]
        df = pd.DataFrame({
            "Class": [CLASS_LABELS[i].title() for i in top3],
            "Confidence (%)": [round(probs[i]*100,2) for i in top3]
        })
        st.dataframe(df, use_container_width=True)

# -----------------------------------
# MODE 2: COMPARE TWO IMAGES
# -----------------------------------
elif mode == "Compare Two Images":
    col1, col2 = st.columns(2)

    with col1:
        file1 = st.file_uploader("📤 Upload First Image", type=["jpg","jpeg","png"], key="first")
        if file1:
            img1 = Image.open(file1)
            st.image(img1, caption="Image 1", use_container_width=True)
            label1, probs1 = predict(img1)

    with col2:
        file2 = st.file_uploader("📤 Upload Second Image", type=["jpg","jpeg","png"], key="second")
        if file2:
            img2 = Image.open(file2)
            st.image(img2, caption="Image 2", use_container_width=True)
            label2, probs2 = predict(img2)

    if file1 and file2:
        st.markdown("### ⚖️ **Comparison Result**")
        st.markdown(f"🔹 **Image 1 → {label1.title()}** ({round(np.max(probs1)*100,2)}% confidence)")
        st.markdown(f"🔸 **Image 2 → {label2.title()}** ({round(np.max(probs2)*100,2)}% confidence)")
        st.write("---")

        table = pd.DataFrame({
            "Image": ["Image 1", "Image 2"],
            "Predicted Class": [label1.title(), label2.title()],
            "Confidence (%)": [round(np.max(probs1)*100,2), round(np.max(probs2)*100,2)]
        })
        st.table(table)

# -----------------------------------
# MODE 3: BATCH PREDICTION
# -----------------------------------
elif mode == "Batch Prediction":
    files = st.file_uploader("📤 Upload Multiple Images", type=["jpg","jpeg","png"], accept_multiple_files=True)

    if files:
        st.markdown("### 📊 **Batch Prediction Results**")
        results = []

        for f in files:
            img = Image.open(f)
            label, probs = predict(img)
            conf = round(np.max(probs)*100,2)

            st.markdown(f"**{f.name} → 🎯 {label.title()}** ({conf}% confidence)")
            results.append([f.name, label.title(), conf])

        st.write("---")
        df = pd.DataFrame(results, columns=["Filename", "Predicted Class", "Confidence (%)"])
        st.dataframe(df, use_container_width=True)
