import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import json
from pathlib import Path
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Report Road Damage",
    page_icon="📢",
    layout="wide"
)

# ----------------------------------
# CONSTANTS
# ----------------------------------
MODEL_PATH = r"C:/Users/badri/Downloads/RoadDamageDetection-main (1)/RoadDamageDetection-main/models/Yolov8nCBAM.pt"

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

DEFAULT_RECIPIENT = "thedynamo456@gmail.com"

REPORTS_DIR = Path("./reports")
IMAGES_DIR = REPORTS_DIR / "images"

REPORTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# ----------------------------------
# CACHE MODEL - LOAD ONCE
# ----------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Initialize session state
if 'captured_frame' not in st.session_state:
    st.session_state['captured_frame'] = None
if 'capture_detections' not in st.session_state:
    st.session_state['capture_detections'] = []
if 'camera_active' not in st.session_state:
    st.session_state['camera_active'] = True

# ----------------------------------
# GLOBAL STATE FOR WEBRTC
# ----------------------------------
class VideoProcessor:
    def __init__(self):
        self.confidence_threshold = 0.3
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated = None
        self.detections = []
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Store latest frame
        with self.lock:
            self.latest_frame = img.copy()
        
        # Run YOLO detection
        results = model.predict(img, conf=self.confidence_threshold, verbose=False)
        
        # Store detections
        detections = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                detections.append({
                    "label": CLASSES[cls],
                    "confidence": float(b.conf[0])
                })
        
        with self.lock:
            self.detections = detections
            
        # Get annotated frame
        annotated_img = results[0].plot()
        
        with self.lock:
            self.latest_annotated = annotated_img.copy()
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# ----------------------------------
# LOCATION DETECTION
# ----------------------------------
def get_location():
    """Get user's approximate location using IP geolocation"""
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        data = response.json()
        return {
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'city': data.get('city'),
            'region': data.get('region'),
            'country': data.get('country_name'),
            'address': f"{data.get('city')}, {data.get('region')}, {data.get('country_name')}"
        }
    except Exception as e:
        return None


# ----------------------------------
# EMAIL UTILITIES
# ----------------------------------
def send_email_report(report_id, detections, location, user_message, pil_image, recipient_email):
    """Send email report with image, location, and user message"""
    
    sender_email = "bhanuhuman86@gmail.com"
    sender_password = "ngby qakk vbti zmga"
    
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Road Damage Report - {report_id}"
        
        detection_list = "<br>".join([f"• {d['label']} (Confidence: {d['confidence']:.2%})" for d in detections])
        
        location_info = "Location not available"
        if location:
            location_info = f"""
            <strong>Address:</strong> {location.get('address', 'N/A')}<br>
            <strong>Coordinates:</strong> {location.get('latitude', 'N/A')}, {location.get('longitude', 'N/A')}<br>
            <strong>City:</strong> {location.get('city', 'N/A')}<br>
            <strong>Region:</strong> {location.get('region', 'N/A')}
            """
        
        user_msg_section = ""
        if user_message and user_message.strip():
            user_msg_section = f"""
            <h3 style="color: #2c5282;">Additional Notes:</h3>
            <p style="background-color: #f7fafc; padding: 10px; border-left: 4px solid #4299e1;">{user_message}</p>
            """
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #e53e3e; border-bottom: 2px solid #e53e3e; padding-bottom: 10px;">
                    🚨 Road Damage Report
                </h2>
                
                <div style="background-color: #fff5f5; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Report ID:</strong> {report_id}</p>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <h3 style="color: #2c5282;">Detected Damages:</h3>
                <div style="background-color: #ebf8ff; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    {detection_list if detections else "No damages detected"}
                </div>
                
                <h3 style="color: #2c5282;">Location Details:</h3>
                <div style="background-color: #f0fff4; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    {location_info}
                </div>
                
                {user_msg_section}
                
                <p style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #718096; font-size: 12px;">
                    This is an automated report from the Road Damage Detection System.
                </p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        image = MIMEImage(img_buffer.read(), name=f"{report_id}.png")
        msg.attach(image)
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"


# ----------------------------------
# UTILITIES
# ----------------------------------
def generate_report_id():
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"RDR-{now}"


def save_report(report_id, metadata, pil_image):
    json_path = REPORTS_DIR / f"{report_id}.json"
    img_path = IMAGES_DIR / f"{report_id}.png"

    pil_image.save(img_path)

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return json_path, img_path


def pil_to_bytes(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------
# UI LAYOUT
# ----------------------------------
st.title("📢 Report Road Damage")

# ----------------------------------
# INPUT SECTION
# ----------------------------------
input_type = st.radio("Input Method", ["Real-time Camera", "Upload Image"], horizontal=True)

if input_type == "Real-time Camera":
    
    # Check if we have a captured frame - if yes, skip camera and show report
    if st.session_state['captured_frame'] is None and st.session_state['camera_active']:
        
        # Camera is active - show live feed
        col_cam, col_controls = st.columns([2, 1])
        
        with col_cam:
            st.subheader("📹 Live Detection")
            
            # Initialize video processor
            ctx = webrtc_streamer(
                key="road-damage-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_processor_factory=VideoProcessor,
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, 
                    "audio": False
                },
                async_processing=True,
            )
        
        with col_controls:
            st.subheader("⚙️ Controls")
            
            score_threshold = st.slider("Confidence", 0.0, 1.0, 0.30, 0.05)
            
            if ctx.video_processor:
                ctx.video_processor.confidence_threshold = score_threshold
                
                st.divider()
                
                # Display current detections
                st.subheader("🔍 Live Detections")
                if ctx.video_processor.detections:
                    for det in ctx.video_processor.detections:
                        st.write(f"**{det['label']}**")
                        st.progress(det['confidence'])
                else:
                    st.info("No damage detected")
                
                st.divider()
                
                # Capture button prominently displayed
                if st.button("📸 CAPTURE & STOP", type="primary", use_container_width=True, key="capture_btn"):
                    if ctx.video_processor.latest_frame is not None:
                        # Capture the frame
                        st.session_state['captured_frame'] = Image.fromarray(
                            cv2.cvtColor(ctx.video_processor.latest_frame, cv2.COLOR_BGR2RGB)
                        )
                        st.session_state['capture_detections'] = ctx.video_processor.detections.copy()
                        st.session_state['camera_active'] = False
                        st.success("✅ Frame captured!")
                        st.rerun()
                    else:
                        st.warning("No frame available")
            else:
                st.warning("Camera not started")
                st.info("Click 'Start' above to begin")

elif input_type == "Upload Image":
    uploaded = st.file_uploader("Upload road image", ["jpg", "jpeg", "png"])
    if uploaded:
        # Load the original image
        original_image = Image.open(uploaded)
        st.session_state['camera_active'] = False
        
        # Run detection immediately
        img_np = np.array(original_image.convert("RGB"))
        results = model.predict(img_np, conf=0.3, verbose=False)
        
        detections = []
        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                detections.append({
                    "label": CLASSES[cls],
                    "confidence": float(b.conf[0])
                })
        
        st.session_state['capture_detections'] = detections
        
        # Get annotated image with bounding boxes
        annotated_img = results[0].plot()
        # Convert BGR to RGB
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        # Store the annotated image instead of original
        st.session_state['captured_frame'] = Image.fromarray(annotated_rgb)


# ----------------------------------
# REPORT GENERATION SECTION (Only shows after capture/upload)
# ----------------------------------
if st.session_state['captured_frame'] is not None:
    st.divider()
    st.header("📋 Create Report")
    
    col_img, col_form = st.columns([1, 1])
    
    with col_img:
        st.subheader("Captured Image")
        st.image(st.session_state['captured_frame'], use_column_width=True)
        
        if st.session_state['capture_detections']:
            st.success(f"✅ {len(st.session_state['capture_detections'])} damage(s) detected")
            with st.expander("View Detections"):
                for det in st.session_state['capture_detections']:
                    st.write(f"• **{det['label']}** - {det['confidence']:.1%}")
        else:
            st.warning("⚠️ No damages detected")
    
    with col_form:
        st.subheader("Report Details")
        
        # Location
        with st.spinner("🌍 Detecting location..."):
            location = get_location()
        
        if location:
            st.success(f"📍 {location['city']}, {location['region']}")
            with st.expander("View location details"):
                st.write(f"**Address:** {location['address']}")
                st.write(f"**Coordinates:** {location['latitude']}, {location['longitude']}")
        
        # Email config
        recipient_email = st.text_input(
            "📧 Recipient Email", 
            placeholder=f"Default: {DEFAULT_RECIPIENT}"
        )
        
        # User message
        user_message = st.text_area(
            "Additional Notes", 
            placeholder="Add context, severity, etc...",
            height=100
        )
        
        # Report ID
        report_id = generate_report_id()
        st.caption(f"Report ID: {report_id}")
        
        st.divider()
        
        # Action buttons
        col_save, col_send = st.columns(2)
        
        with col_save:
            if st.button("💾 Save Locally", use_container_width=True):
                metadata = {
                    "report_id": report_id,
                    "timestamp": datetime.now().isoformat(),
                    "detections": st.session_state['capture_detections'],
                    "location": location,
                    "user_message": user_message if user_message.strip() else None
                }
                
                json_path, img_path = save_report(report_id, metadata, st.session_state['captured_frame'])
                
                st.success("✅ Saved!")
                st.caption(f"📄 {json_path}")
                st.caption(f"🖼️ {img_path}")
        
        with col_send:
            if st.button("📧 Send Email", type="primary", use_container_width=True):
                email_to_use = recipient_email.strip() if recipient_email and recipient_email.strip() else DEFAULT_RECIPIENT
                
                with st.spinner("Sending..."):
                    success, message = send_email_report(
                        report_id, 
                        st.session_state['capture_detections'], 
                        location, 
                        user_message, 
                        st.session_state['captured_frame'], 
                        email_to_use
                    )
                
                if success:
                    st.success("✅ Email sent!")
                    st.balloons()
                else:
                    st.error(f"❌ {message}")
        
        # Download buttons
        st.divider()
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            metadata = {
                "report_id": report_id,
                "timestamp": datetime.now().isoformat(),
                "detections": st.session_state['capture_detections'],
                "location": location,
                "user_message": user_message if user_message.strip() else None
            }
            
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(metadata, indent=4),
                file_name=f"{report_id}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_dl2:
            st.download_button(
                "⬇️ Download Image",
                data=pil_to_bytes(st.session_state['captured_frame']),
                file_name=f"{report_id}.png",
                mime="image/png",
                use_container_width=True
            )
        
        st.divider()
        
        # Clear captured frame and restart
        if st.button("🔄 Capture New Image", use_container_width=True):
            st.session_state['captured_frame'] = None
            st.session_state['capture_detections'] = []
            st.session_state['camera_active'] = True
            st.rerun()

else:
    if input_type == "Real-time Camera":
        st.info("👆 Click 'Start' on the camera above, then click 'CAPTURE & STOP' when ready")
    else:
        st.info("👆 Upload an image to begin")