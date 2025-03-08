import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import warnings
import base64
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Streamlit page config
st.set_page_config(page_title="VisorAI", layout="wide", page_icon="assets/icon.png")

# Function to play sound using HTML5 audio (for Streamlit Cloud compatibility)
def play_sound():
    audio_file = "assets/helmet_detected.mp3"
    if os.path.exists(audio_file):
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()

        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# Function to encode an image to Base64 for embedding
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

# Load and display logo
logo_base64 = get_base64_image("assets/icon.png")
if logo_base64:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; padding-top: 50px;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 100px; height: auto; margin-right: 10px;">
            <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Load YOLO model (cached to improve performance)
@st.cache_resource
def load_model():
    return YOLO("helmet_detector.pt")

model = load_model()

# Create tabs
detect, model_info = st.tabs(["Detection", "Model Information"])

# Store previous detections to avoid repeated alerts
previous_detections = set()

# Function to check new helmet detections and trigger sound
def check_new_helmet_detections(results):
    global previous_detections
    new_detections = set()

    for detection in results[0].boxes:
        class_id = int(detection.cls[0])
        confidence = detection.conf[0].item()

        if confidence > 0.5:
            new_detections.add(class_id)

    if new_detections - previous_detections:
        play_sound()

    previous_detections.update(new_detections)

# Process image for helmet detection
def process_image(image):
    image = np.array(image)
    results = model(image, verbose=False)
    check_new_helmet_detections(results)
    detected_img = results[0].plot(conf=True)
    return cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

# Process video for helmet detection
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        check_new_helmet_detections(results)

        detected_frame = results[0].plot(conf=True)
        detected_frame = cv2.cvtColor(np.array(detected_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        stframe.image(detected_frame, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

# File uploader (supports both image and video)
uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Detection tab logic
with detect:
    if uploaded_file:
        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            image = Image.open(uploaded_file)
            with st.spinner("Processing image..."):
                detected_img = process_image(image)
            st.image(detected_img, caption="Detected Image", use_container_width=True)

        elif file_type == "video":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            with st.spinner("Processing video..."):
                process_video(temp_file_path)

            os.remove(temp_file_path)
    else:
        if os.path.exists("assets/vid.mp4"):
            st.video("assets/vid.mp4")
        else:
            st.warning("No video found in assets folder.")

