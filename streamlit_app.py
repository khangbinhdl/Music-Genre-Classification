import streamlit as st

from src.inference.predictor import MLPGenrePredictor, MusicGenrePredictor

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")
st.title("Music Genre Classification")
st.caption("Upload WAV file, extract 3-second features with librosa, then predict genre.")

@st.cache_resource
def load_ml_predictor():
    return MusicGenrePredictor(model_dir="models/machine_learning")


@st.cache_resource
def load_mlp_predictor():
    return MLPGenrePredictor(model_dir="models/deep_learning")

model_option = st.radio(
    "Select inference model",
    options=["Best Model (Machine Learning)", "MLP (Deep Learning)"],
    horizontal=True,
)

if model_option.startswith("MLP"):
    try:
        predictor = load_mlp_predictor()
    except Exception as exc:
        st.warning("MLP is unavailable in current environment. Falling back to Best Model (Machine Learning).")
        st.caption(f"Reason: {exc}")
        predictor = load_ml_predictor()
        model_option = "Best Model (Machine Learning)"
else:
    predictor = load_ml_predictor()

uploaded = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded is not None:
    st.audio(uploaded.getvalue(), format="audio/wav")

    if st.button("Predict genre", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0, text="Starting inference...")

        def on_progress(progress: float, message: str) -> None:
            pct = max(0, min(100, int(progress * 100)))
            progress_text.text(message)
            progress_bar.progress(pct, text=f"{message} ({pct}%)")

        with st.spinner("Extracting features and running inference..."):
            result = predictor.predict_from_bytes(
                uploaded.getvalue(),
                progress_callback=on_progress,
            )

        progress_bar.progress(100, text="Inference completed (100%)")
        progress_text.empty()

        st.success(f"Predicted genre ({model_option}): {result['predicted_genre']}")

        probs = result.get("probabilities")
        if probs:
            st.subheader("Probabilities")
            sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
            st.bar_chart(sorted_probs)
            st.json(sorted_probs)
