import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

st.set_page_config(page_title="Realtime Detection (Offline)", page_icon="📷")

MODEL_PATH = r"C:/Users/badri/Downloads/RoadDamageDetection-main (1)/RoadDamageDetection-main/models/Yolov8nCBAM.pt"

# ----------------------------
# Initialize session state
# ----------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'camera' not in st.session_state:
    st.session_state.camera = None

# ----------------------------
# Cache model (Loads ONCE)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ----------------------------
# Camera management functions
# ----------------------------
def start_camera(device_index):
    """Initialize and start the camera"""
    if st.session_state.camera is not None:
        st.session_state.camera.release()
    
    cam = cv2.VideoCapture(device_index)
    cam.set(3, 1280)
    cam.set(4, 720)
    st.session_state.camera = cam
    return cam

def stop_camera():
    """Release the camera properly"""
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None

st.title("Road Damage Detection - Offline Realtime")

device_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], 0)
score_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3)
show_table = st.sidebar.checkbox("Show Predictions")

# Button callbacks to update session state
if st.button("Start"):
    st.session_state.running = True
    start_camera(device_index)
    st.rerun()

if st.button("Stop"):
    st.session_state.running = False
    stop_camera()
    st.rerun()

frame_area = st.empty()
table_area = st.empty()

# Use session state for running status
if st.session_state.running:
    cam = st.session_state.camera
    
    if cam is None or not cam.isOpened():
        st.error("Camera not available")
        st.session_state.running = False
        st.rerun()
    
    while st.session_state.running:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to read from camera")
            st.session_state.running = False
            stop_camera()
            break

        results = model.predict(frame, conf=score_threshold, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        frame_area.image(annotated, channels="RGB", use_container_width=True)

        if show_table:
            detections = []
            for b in results[0].boxes:
                detections.append({
                    "Class": int(b.cls[0]),
                    "Conf": float(b.conf[0])
                })
            if detections:
                table_area.table(detections)
            else:
                table_area.empty()

        time.sleep(0.01)
else:
    st.info("Click 'Start' to begin detection")
    # Clean up message when stopped
    if st.session_state.camera is None:
        frame_area.empty()
        table_area.empty()