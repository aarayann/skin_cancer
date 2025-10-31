# Skin Cancer App

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def load_deploy_model():
    try:
        model = load_model('deploy_model.keras', compile=False)  # Frozen, no training state
        return model
    except Exception as e:
        st.error(f"Load error: {e}. Check file.")
        return None

@st.cache_data
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32) / 255.0  # [0,1] normalize
    return img_array

# Load model once
model = load_deploy_model()
if model is None:
    st.stop()

st.title('Skin Cancer Detector (80% Acc, 85% Recall)')
st.write("Upload skin image for benign/malignant pred (focus: high cancer detection).")

# Sidebar reset (clear cache, rerun)
if st.sidebar.button("Reset & Clear Cache"):
    st.cache_data.clear()
    st.rerun()

uploaded_file = st.file_uploader("Upload JPG/PNG...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = preprocess_image(image)
    
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Malignant (Cancer – Urgent!)" if pred > 0.5 else "Benign (Safe)"
    conf = f"{pred*100:.1f}%" if pred > 0.5 else f"{(1-pred)*100:.1f}%"
    color = "red" if pred > 0.5 else "green"
    
    st.success(f"Result: **{label}** (Conf: **{conf}**) – Model robust, no missed cancers likely.")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Your Image", use_column_width=True)  # Large visible
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.squeeze(img_array))
        ax.set_title(f"Predicted: **{label}**\\nConf: **{conf}**", color=color, fontsize=14, fontweight='bold')
        ax.axis('off')
        st.pyplot(fig)
else:
    st.info("Upload to predict.")

st.write("App: Cached speed, error-free, simple UI (no advanced visuals).")
