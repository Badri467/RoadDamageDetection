import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# USE YOUR LOCAL YOLO MODEL (Instead of downloading)
# ---------------------------------------------------------
MODEL_LOCAL_PATH = r"C:/Users/badri/Downloads/RoadDamageDetection-main (1)/RoadDamageDetection-main/models/Yolov8nCBAM.pt"

# Load once per session
cache_key = "yolov8cbam"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

# Class Labels
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

st.title("Road Damage Detection - Image")
st.write("Detect road damage by uploading an image.")

# Upload Image
image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

# Default threshold changed to 0.30
score_threshold = st.slider(
    "Confidence Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.30,    # <<< SET DEFAULT TO 30%
    step=0.05
)

st.write("Lower the threshold to detect more damage, increase to reduce false positives.")

if image_file is not None:
    
    # Load the image
    image = Image.open(image_file)

    col1, col2 = st.columns(2)

    # Convert image to numpy
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    # Resize for YOLO
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)

    # YOLO prediction
    results = net.predict(image_resized, conf=score_threshold)

    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    # Annotated output
    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Original Image
    with col1:
        st.write("#### Image")
        st.image(_image)

    # Predicted Image
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Download Button
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
