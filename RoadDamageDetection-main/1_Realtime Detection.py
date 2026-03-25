import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

st.set_page_config(page_title="Realtime Detection (Offline)", page_icon="📷")

MODEL_PATH = r"C:/Users/badri/Downloads/RoadDamageDetection-main (1)/RoadDamageDetection-main/models/YOLOv8_Small_RDD.pt"

# ----------------------------
# Initialize session state
# ----------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = deque(maxlen=5)  # Keep last 5 frames

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
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
    st.session_state.camera = cam
    return cam

def stop_camera():
    """Release the camera properly"""
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    st.session_state.detection_history.clear()

def stabilize_detections(current_detections, history, min_appearances=2):
    """Only show detections that appear in multiple consecutive frames"""
    # Add current detections to history
    history.append(current_detections)
    
    # Count how many times each class appears in recent frames
    class_counts = {}
    for frame_dets in history:
        for det in frame_dets:
            cls = det['Class']
            if cls not in class_counts:
                class_counts[cls] = {'count': 0, 'max_conf': 0}
            class_counts[cls]['count'] += 1
            class_counts[cls]['max_conf'] = max(class_counts[cls]['max_conf'], det['Conf'])
    
    # Only return detections that appeared multiple times
    stable_detections = []
    for cls, info in class_counts.items():
        if info['count'] >= min_appearances:
            stable_detections.append({
                'Class': cls,
                'Conf': info['max_conf'],
                'Stability': f"{info['count']}/{len(history)}"
            })
    
    return stable_detections

st.title("🛣️ Road Damage Detection - Offline Realtime")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
device_index = st.sidebar.selectbox("📹 Select Camera", [0, 1, 2], 0)
score_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.30, 0.05)
iou_threshold = st.sidebar.slider("📊 IoU Threshold", 0.0, 1.0, 0.45, 0.05)
show_table = st.sidebar.checkbox("📋 Show Predictions", value=True)
stabilization = st.sidebar.checkbox("🔒 Stabilize Detections", value=True, 
                                    help="Only show detections that appear in multiple frames")
min_appearances = st.sidebar.slider("🎚️ Min Frames for Stability", 1, 5, 2, 1,
                                   help="How many frames a detection must appear to be shown")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:**\n- Lower confidence = more detections\n- Enable stabilization to reduce flickering\n- Increase min frames for more stable results")

# Control buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("▶️ Start Detection", use_container_width=True)
with col2:
    stop_btn = st.button("⏹️ Stop Detection", use_container_width=True)

if start_btn:
    st.session_state.running = True
    st.session_state.detection_history.clear()
    start_camera(device_index)
    st.rerun()

if stop_btn:
    st.session_state.running = False
    stop_camera()
    st.rerun()

# Display areas
frame_area = st.empty()
stats_col1, stats_col2, stats_col3 = st.columns(3)
table_area = st.empty()

# Main detection loop
if st.session_state.running:
    cam = st.session_state.camera
    
    if cam is None or not cam.isOpened():
        st.error("❌ Camera not available")
        st.session_state.running = False
        st.rerun()
    
    fps_counter = 0
    start_time = time.time()
    
    while st.session_state.running:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Failed to read from camera")
            st.session_state.running = False
            stop_camera()
            break

        # Run detection with optimized parameters
        results = model.predict(
            frame, 
            conf=score_threshold,
            iou=iou_threshold,
            verbose=False,
            device='cpu',  # Change to 'cuda' or '0' if you have GPU
            half=False  # Set to True if using GPU for faster inference
        )
        
        # Get detections
        current_detections = []
        for b in results[0].boxes:
            current_detections.append({
                "Class": int(b.cls[0]),
                "Conf": float(b.conf[0])
            })
        
        # Apply stabilization if enabled
        if stabilization:
            display_detections = stabilize_detections(
                current_detections, 
                st.session_state.detection_history,
                min_appearances
            )
        else:
            display_detections = current_detections
        
        # Draw annotations
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Add detection count overlay
        cv2.putText(annotated, f"Detections: {len(display_detections)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame_area.image(annotated, channels="RGB", use_container_width=True)
        
        # Calculate FPS
        fps_counter += 1
        elapsed = time.time() - start_time
        if elapsed > 1.0:
            fps = fps_counter / elapsed
            with stats_col1:
                st.metric("🎥 FPS", f"{fps:.1f}")
            with stats_col2:
                st.metric("🔍 Detections", len(display_detections))
            with stats_col3:
                st.metric("📊 Raw Detections", len(current_detections))
            fps_counter = 0
            start_time = time.time()
        
        # Show detection table
        if show_table and display_detections:
            table_area.table(display_detections)
        elif show_table:
            table_area.info("No stable detections")
        
        # Small delay to prevent UI blocking
        time.sleep(0.001)
        
else:
    st.info("👆 Click 'Start Detection' to begin")
    frame_area.empty()
    table_area.empty()