import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# USE YOUR LOCAL MODEL HERE
# ---------------------------------------------------------
MODEL_LOCAL_PATH = r"C:/Users/badri/Downloads/RoadDamageDetection-main (1)/RoadDamageDetection-main/models/Yolov8nCBAM.pt"

# Load model once per session
cache_key = "yolov8cbam_video"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Create temp folder
if not os.path.exists('./temp'):
    os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Detect running mode
if 'processing_button' in st.session_state and st.session_state.processing_button:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

def write_bytesio_to_file(filename, bytesio):
    """Save uploaded video to disk."""
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    write_bytesio_to_file(temp_file_input, video_file)

    videoCapture = cv2.VideoCapture(temp_file_input)

    if not videoCapture.isOpened():
        st.error("Error opening the video file")
        return
    
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = frame_count / fps
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    st.write(f"Video Duration: {minutes}:{seconds:02d}")
    st.write("Width, Height, FPS:", width, height, fps)

    inferenceBarText = "Processing video... please wait."
    inferenceBar = st.progress(0, text=inferenceBarText)

    imageLocation = st.empty()

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2writer = cv2.VideoWriter(temp_file_infer, fourcc, fps, (width, height))

    frame_counter = 0
    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(frame_rgb, (640, 640))
        results = net.predict(resized, conf=score_threshold)

        annotated = results[0].plot()
        annotated_resized = cv2.resize(annotated, (width, height))

        imageLocation.image(annotated_resized)

        out_frame = cv2.cvtColor(annotated_resized, cv2.COLOR_RGB2BGR)
        cv2writer.write(out_frame)

        frame_counter += 1
        inferenceBar.progress(frame_counter / frame_count, text=inferenceBarText)

    videoCapture.release()
    cv2writer.release()

    inferenceBar.empty()
    st.success("Video Processed!")

    col1, col2 = st.columns(2)
    with col1:
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4"
            )

    with col2:
        if st.button("Restart App", use_container_width=True):
            st.rerun()

st.title("Road Damage Detection - Video")
st.write("Upload a video and run detection using YOLOv8 + CBAM.")

video_file = st.file_uploader(
    "Upload Video (.mp4)", 
    type=".mp4", 
    disabled=st.session_state.runningInference
)

score_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.30,      # <<< DEFAULT SET TO 0.30
    step=0.05,
    disabled=st.session_state.runningInference
)

st.caption("Lower threshold for more detections. Increase to reduce false positives.")

if video_file is not None:
    if st.button(
        "Process Video",
        use_container_width=True,
        disabled=st.session_state.runningInference,
        key="processing_button"
    ):
        st.warning("Processing Video: " + video_file.name)
        processVideo(video_file, score_threshold)
