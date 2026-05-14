import streamlit as st
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🖼️",
    layout="centered",
)

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
    "cat": "🐱", "deer": "🦌", "dog": "🐶",
    "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚛",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_cnn_cifar10.keras")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "hyperparameter_results_cifar10.json")


@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("ℹ️ Acerca del modelo")
st.sidebar.markdown(
    "**Dataset:** CIFAR-10  \n"
    "**Arquitectura:** CNN personalizada  \n"
    "**Optimización:** Grid Search  \n"
)

if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Resultados en test:**")
    st.sidebar.metric("Test Accuracy", f"{results.get('test_accuracy', 0):.2%}")
    st.sidebar.metric("Test Loss", f"{results.get('test_loss', 0):.4f}")
    hp = results.get("best_hyperparameters", {})
    if hp:
        st.sidebar.markdown("**Mejores hiperparámetros:**")
        for k, v in hp.items():
            st.sidebar.markdown(f"- `{k}`: `{v}`")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🖼️ Clasificador CIFAR-10")
st.markdown(
    "Sube una imagen y el modelo CNN la clasificará en una de las **10 categorías** de CIFAR-10."
)

uploaded = st.file_uploader(
    "Selecciona una imagen (PNG, JPG, JPEG, WEBP)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded is not None:
    img = Image.open(uploaded)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(img, caption="Imagen original", use_column_width=True)

    with col2:
        with st.spinner("Cargando modelo y clasificando..."):
            model = load_model()
            x = preprocess(img)
            probs = model.predict(x, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        st.markdown(f"### {CLASS_EMOJI[pred_class]} {pred_class.upper()}")
        st.markdown(f"**Confianza:** {confidence:.1%}")
        st.progress(confidence)

    st.markdown("---")
    st.markdown("#### Probabilidades por clase")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        label = f"{CLASS_EMOJI[CLASSES[i]]} {CLASSES[i]}"
        st.markdown(f"**{label}**")
        st.progress(float(probs[i]), text=f"{probs[i]:.1%}")

else:
    st.info("👆 Sube una imagen para empezar.")
    st.markdown(
        "**Clases disponibles:** " +
        " · ".join(f"{CLASS_EMOJI[c]} {c}" for c in CLASSES)
    )
