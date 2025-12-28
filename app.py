import streamlit as st
import os
import time

# =========================================================
# PAGE CONFIG (must be first)
# =========================================================
st.set_page_config(
    page_title="Solar Power Forecaster",
    layout="wide"
)

# =========================================================
# CORE IMPORTS
# =========================================================
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_model_paths(forecast_type):
    """
    Returns file paths and sequence length based on forecast horizon.
    """
    base_dir = os.getcwd()

    if forecast_type == "15 Minute Forecast":
        artifact_dir = os.path.join(base_dir, "model_artifacts_15min")
        return {
            "model_path": os.path.join(artifact_dir, "lstm_model_15min.h5"),
            "feature_scaler": os.path.join(artifact_dir, "feature_scaler_15min.pkl"),
            "target_scaler": os.path.join(artifact_dir, "target_scaler_15min.pkl"),
            "sequence_length": 15
        }

    if forecast_type == "1 Hour Forecast":
        artifact_dir = os.path.join(base_dir, "model_artifacts_1hr")
        return {
            "model_path": os.path.join(artifact_dir, "lstm_model_1hr.h5"),
            "feature_scaler": os.path.join(artifact_dir, "feature_scaler_1hr.pkl"),
            "target_scaler": os.path.join(artifact_dir, "target_scaler_1hr.pkl"),
            "sequence_length": 60
        }

    return None


def load_model_and_scalers(paths):
    """
    Loads TensorFlow model and scalers lazily.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        model = load_model(paths["model_path"])
        feature_scaler = joblib.load(paths["feature_scaler"])
        target_scaler = joblib.load(paths["target_scaler"])

        return model, feature_scaler, target_scaler

    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

    return None, None, None


def run_prediction(model, feature_scaler, target_scaler, data, seq_len):
    """
    Prepares data and runs model inference.
    """
    required_features = ['Irradiation', 'Temp', 'Wind', 'Humidity', 'Barometer']

    missing_cols = [col for col in required_features if col not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None

    if len(data) < seq_len:
        st.error(f"At least {seq_len} rows are required for this forecast.")
        return None

    recent_data = data.iloc[-seq_len:][required_features]
    scaled_input = feature_scaler.transform(recent_data)
    X = scaled_input.reshape(1, seq_len, len(required_features))

    scaled_prediction = model.predict(X)
    prediction_mw = target_scaler.inverse_transform(scaled_prediction)

    return prediction_mw.flatten()


# =========================================================
# MAIN APPLICATION UI
# =========================================================

st.title("Solar Power Forecaster")
st.markdown("### Intelligent Energy Prediction System")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Configuration")

forecast_option = st.sidebar.selectbox(
    "Select Forecast Horizon",
    ["15 Minute Forecast", "1 Hour Forecast"]
)

required_rows = 15 if "15" in forecast_option else 60
st.sidebar.info(
    f"""
    Selected Mode: {forecast_option}  
    Minimum Required Rows: {required_rows}
    """
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Input Excel File", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload an Excel file to begin.")
    st.stop()

# ---------------- READ DATA ----------------
try:
    data = pd.read_excel(uploaded_file)

    if "Time" in data.columns:
        data["Time"] = pd.to_datetime(data["Time"])

    st.success("Data uploaded successfully")

    with st.expander("Preview Uploaded Data"):
        st.dataframe(data.tail(10))

except Exception as e:
    st.error(f"Failed to read Excel file: {e}")
    st.stop()

# ---------------- PREDICTION ----------------
if st.button("Generate Forecast"):

    paths = get_model_paths(forecast_option)

    progress = st.progress(0)
    status = st.empty()

    status.text("Initializing model...")
    model, feat_scaler, tgt_scaler = load_model_and_scalers(paths)
    progress.progress(50)

    if model is None:
        st.stop()

    status.text("Running prediction...")
    predictions = run_prediction(
        model,
        feat_scaler,
        tgt_scaler,
        data,
        paths["sequence_length"]
    )
    progress.progress(100)

    if predictions is None:
        st.stop()

    time.sleep(0.5)
    status.empty()
    progress.empty()

    # ---------------- RESULTS ----------------
    st.divider()
    st.subheader(f"{forecast_option} Results")

    last_time = data["Time"].iloc[-1] if "Time" in data.columns else pd.Timestamp.now()
    future_times = [
        last_time + pd.Timedelta(minutes=i + 1)
        for i in range(len(predictions))
    ]

    results_df = pd.DataFrame({
        "Time": future_times,
        "Predicted Power (MW)": predictions
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df["Time"],
        y=results_df["Predicted Power (MW)"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#FFA500", width=3)
    ))

    fig.update_layout(
        title="Predicted Solar Power Output",
        xaxis_title="Time",
        yaxis_title="Megawatts (MW)",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(results_df, height=300)

    with col2:
        st.success(f"Average Output: {results_df['Predicted Power (MW)'].mean():.2f} MW")
        st.info(f"Maximum Output: {results_df['Predicted Power (MW)'].max():.2f} MW")

        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            csv_data,
            file_name="solar_forecast.csv",
            mime="text/csv"
        )
