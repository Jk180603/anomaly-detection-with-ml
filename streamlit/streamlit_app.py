import sys
import os

sys.path.append(os.path.abspath("src"))

import json
import numpy as np
import pandas as pd
import streamlit as st

from predict import predict_anomaly


st.set_page_config(
    page_title="Industrial Anomaly Detection",
    page_icon="⚙️",
    layout="centered",
)

st.title("⚙️ Industrial Sensor Anomaly Detection")
st.write("LSTM Autoencoder-based anomaly detection for NASA turbofan engine sensor data.")

st.divider()

option = st.radio(
    "Choose input method:",
    ["Use sample sequence", "Paste JSON sequence"],
)

sequence = None

if option == "Use sample sequence":
    sample_index = st.number_input(
        "Sample index",
        min_value=0,
        max_value=1000,
        value=0,
        step=1,
    )

    if st.button("Load sample"):
        sequences = np.load("data/processed/train_sequences.npy")
        sequence = sequences[int(sample_index)]
        st.success(f"Loaded sample sequence index: {sample_index}")
        st.write("Sequence shape:", sequence.shape)

elif option == "Paste JSON sequence":
    json_input = st.text_area(
        "Paste JSON here",
        height=300,
        placeholder='{"sequence": [[...], [...]]}',
    )

    if json_input:
        try:
            parsed = json.loads(json_input)
            sequence = np.array(parsed["sequence"])
            st.success("JSON parsed successfully")
            st.write("Sequence shape:", sequence.shape)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

if sequence is not None:
    st.divider()

    if st.button("Predict Anomaly"):
        result = predict_anomaly(sequence)

        status = result["status"]
        error = result["reconstruction_error"]
        threshold = result["threshold"]

        if result["is_anomaly"]:
            st.error(f"🚨 Status: {status}")
        else:
            st.success(f"✅ Status: {status}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Reconstruction Error", round(error, 6))

        with col2:
            st.metric("Anomaly Threshold", round(threshold, 6))

        st.json(result)

        df = pd.DataFrame(sequence)
        st.line_chart(df)