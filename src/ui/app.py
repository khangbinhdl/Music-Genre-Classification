from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="GTZAN Music Genre Classifier", page_icon="🎵", layout="centered")
st.title("🎵 GTZAN Music Genre Classifier")
st.caption("Upload WAV, chọn model ML/DL, trích xuất đặc trưng 3 giây đầu và xem xác suất dự đoán.")

@st.cache_data(ttl=30)
def fetch_models(api_url: str) -> dict:
    response = requests.get(f"{api_url}/models", timeout=10)
    response.raise_for_status()
    return response.json()

with st.sidebar:
    st.header("API")
    api_url = st.text_input("FastAPI URL", value=API_URL)

try:
    payload = fetch_models(api_url)
    available = payload["models"]
except Exception as exc:
    st.error(f"Không kết nối được API hoặc chưa có model: {exc}")
    st.stop()

model_type_label = st.radio("Loại model", ["Machine learning", "Deep learning"], horizontal=True)
model_type = "machine_learning" if model_type_label == "Machine learning" else "deep_learning"
model_options = payload.get("model_options", {}).get(model_type, [])
if model_options:
    model_name_by_display = {item["display_name"]: item["name"] for item in model_options}
    model_display_names = list(model_name_by_display.keys())
else:
    model_names = available.get(model_type, [])
    model_name_by_display = {name: name for name in model_names}
    model_display_names = list(model_name_by_display.keys())

if not model_display_names:
    st.warning(f"Chưa có model cho nhóm {model_type_label}. Hãy chạy `make train` trước.")
    st.stop()

selected_display_name = st.selectbox("Model", model_display_names)
model_name = model_name_by_display[selected_display_name]
uploaded_file = st.file_uploader("Upload file WAV", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

if st.button("Predict", type="primary", disabled=uploaded_file is None):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
    data = {"model_type": model_type, "model_name": model_name}

    with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
        try:
            response = requests.post(f"{api_url}/predict", data=data, files=files, timeout=60)
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            st.error(f"Predict thất bại: {exc}")
            st.stop()

    top = result["top_prediction"]
    st.success(f"Dự đoán cao nhất: {top['genre']} ({top['probability']:.2%})")

    df = pd.DataFrame(result["probabilities"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("genre")["probability"])
