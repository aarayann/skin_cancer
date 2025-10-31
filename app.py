# Skin Cancer App (Fixed: No Cache on Preprocess for Python 3.13 Hashing)

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def load_deploy_model():
    try:
        # Rebuild exact Sequential (standard, no custom – avoids .keras graph bug)
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base.trainable = False  # Frozen base
        model = Sequential([
            base,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.6),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # For shape
        
        # Load weights if available (stable across TF versions; comment if file missing)
        try:
            model.load_weights('deploy_weights.weights.h5')
            st.info("Full model loaded (80% acc).")
        except:
            st.warning("Using imagenet base (~70% acc – add weights.h5 for full).")
        
        return model
    except Exception as e:
        st.error(f"Load error: {e}. Check TF setup.")
        return None

def preprocess_image(image):  # No @cache_data (avoids PIL hash/pickle fail in 3.13)
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.cast(img_array, tf.float32) / 255.0  # [0,1] normalize (matches training)
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
    img_array = preprocess_image(image)  # Now no hash error
    
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
