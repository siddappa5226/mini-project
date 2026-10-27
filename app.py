from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from image_processing import preprocess_image, extract_features

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "damage_model.joblib"

st.set_page_config(page_title="Corneal Preservation AI", page_icon="🔬", layout="wide")
st.title("Corneal Preservation AI Dashboard")
st.caption("Research prototype • Not for clinical diagnosis or transplant decisions")

st.warning(
    "A predicted grade is only as reliable as the labelled training data. "
    "This prototype is not clinically validated and does not determine whether a cornea is suitable for transplantation."
)

left, right = st.columns([1.2, 1])
with left:
    st.subheader("1. Microscopy image")
    uploaded = st.file_uploader("Upload a corneal endothelial microscopy image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        image = np.array(pil_image)
        processed, gray = preprocess_image(image)
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Original image", use_container_width=True)
        with c2:
            st.image(processed, caption="Preprocessed image", use_container_width=True)

        features = extract_features(gray)
        st.subheader("2. Extracted image features")
        feature_df = pd.DataFrame([features]).T.reset_index()
        feature_df.columns = ["Feature", "Value"]
        st.dataframe(feature_df, hide_index=True, use_container_width=True)

        st.subheader("3. AI damage-grade estimate")
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                row = pd.DataFrame([features])[model.feature_names_in_]
                prediction = model.predict(row)[0]
                st.metric("Model output", str(prediction))
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(row)[0]
                    probability_df = pd.DataFrame({"Grade": model.classes_, "Model score": probs})
                    st.dataframe(probability_df, hide_index=True, use_container_width=True)
                st.caption("Model scores are not calibrated clinical probabilities.")
            except Exception as exc:
                st.error(f"Could not run the saved model: {exc}")
                st.info("Retrain the model after checking dataset/labels.csv.")
        else:
            st.info(
                "No trained model found. Add expert-labelled images to dataset/images, "
                "complete dataset/labels.csv, and run `python train_model.py`."
            )
    else:
        st.info("Upload an image to begin. The app will not invent a damage grade without a trained model.")

with right:
    st.subheader("4. Storage temperature")
    mode = st.radio("Temperature source", ["Demo / simulated", "Serial (Arduino)"], horizontal=False)
    if mode == "Demo / simulated":
        temp = st.number_input("Demo temperature (°C)", value=4.1, step=0.1, format="%.2f")
        threshold_low = st.number_input("Lower configured limit (°C)", value=2.0, step=0.5)
        threshold_high = st.number_input("Upper configured limit (°C)", value=8.0, step=0.5)
        st.caption("Demo limits are placeholders. Replace them with limits approved for your lab protocol.")
    else:
        import serial
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
        port = st.selectbox("Arduino COM port", ports if ports else ["No serial ports found"])
        baud = st.selectbox("Baud rate", [9600, 115200], index=0)
        threshold_low = st.number_input("Lower configured limit (°C)", value=2.0, step=0.5)
        threshold_high = st.number_input("Upper configured limit (°C)", value=8.0, step=0.5)
        temp = None
        if st.button("Read latest Arduino temperature"):
            if port == "No serial ports found":
                st.error("No serial port detected. Connect Arduino and check its driver/COM port.")
            else:
                try:
                    with serial.Serial(port, baud, timeout=3) as ser:
                        ser.reset_input_buffer()
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                    temp = float(line)
                    st.session_state["last_temp"] = temp
                except Exception as exc:
                    st.error(f"Could not read temperature: {exc}")
        temp = st.session_state.get("last_temp", temp)

    if temp is not None:
        st.metric("Latest temperature", f"{temp:.2f} °C")
        if temp < threshold_low or temp > threshold_high:
            st.error("Temperature is outside the configured demo/lab limits.")
        else:
            st.success("Temperature is within the configured limits.")
        st.caption("The displayed status only compares the reading with your configured limits.")
    else:
        st.info("Connect Arduino or use demo mode to view a temperature.")

st.divider()
st.subheader("5. Interpretation")
st.write(
    "The software shows image-derived features, an optional trained-model output, and a temperature reading. "
    "To study a relationship between storage conditions and damage, link each image to its matching sample/time-period "
    "and analyse the paired records. Temperature alone cannot establish cellular damage."
)
st.caption("Recovery prediction is not implemented in this starter because it needs separate, validated recovery labels and a defined target.")
