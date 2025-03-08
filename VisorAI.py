import streamlit as st
import tempfile
import cv2
import numpy as np
import pygame
from PIL import Image
from ultralytics import YOLO
import os
import warnings
import base64
import time

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="VisorAI", layout="wide", page_icon='assets/icon.png')

# Initialize pygame for playing sound
pygame.mixer.init()
helmet_sound = "assets/helmet_detected.mp3"  # Path to the alert sound file

def play_sound():
    if not pygame.mixer.music.get_busy():  # Only play if no sound is currently playing
        pygame.mixer.music.load(helmet_sound)
        pygame.mixer.music.play()

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Convert logo to base64
logo_base64 = get_base64_image('assets/icon.png')

# Streamlit UI: Logo & Title
st.markdown(
    f"""
    <div style="display: flex; align-items: center; padding-top: 50px;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; height: auto; margin-right: 10px;">
        <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .stTabs [role="tablist"] button {
        font-size: 1.2rem;
        padding: 12px 24px;
        margin-right: 10px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("helmet_detector.pt")  # Load YOLO model once

model = load_model()

# Detection UI
detect, model_info = st.tabs(['Detection', 'Model Information'])

# Function to check for helmet detections and play sound per frame (only if not playing)
def check_new_helmet_detections(results):
    helmet_detected = False  # Track if at least one helmet is detected in the frame

    for detection in results[0].boxes:
        class_id = int(detection.cls[0])  # Class ID of detected object
        confidence = detection.conf[0].item()  # Confidence score
        
        if confidence > 0.5:  # Adjust confidence threshold as needed
            helmet_detected = True
            break  # No need to check further; at least one helmet is detected

    if helmet_detected:
        play_sound()  # Play sound only if not currently playing

# Function to process images
def process_image(image):
    image = np.array(image)  # Convert to NumPy array
    results = model(image, verbose=False)  # Run YOLO inference

    check_new_helmet_detections(results)  # Check for new detections

    detected_img = results[0].plot(conf=True)  # Draw detections
    detected_img = cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)  # Convert to BGR
    return detected_img

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()  # Streamlit frame for video display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends

        results = model(frame, verbose=False)  # Run YOLO detection
        check_new_helmet_detections(results)  # Check for detections in the current frame

        detected_frame = results[0].plot(conf=True)  # Draw detections
        detected_frame = cv2.cvtColor(np.array(detected_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        stframe.image(detected_frame, channels="BGR", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

# Sidebar file upload
uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Main UI logic
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

            os.remove(temp_file_path)  # Cleanup temporary file
    else:
        st.video("assets/vid.mp4")  # Default video

# Footer Section
footer = f"""
<hr>
<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; padding: 10px 0;">
  <div style="flex-grow: 1; text-align: left;">
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 100px; margin-right: 10px;">
        <h1 style="margin: 0;">Visor<span style="color:#4CAF50;">AI</span></h1>
    </div>
  </div>
  <!-- Copyright -->
  <div style="flex-grow: 1; text-align: center;">
    <span>Copyright 2024 | All Rights Reserved</span>
  </div>
  <!-- Social media icons -->
  <div style="flex-grow: 1; text-align: right;">
    <a href="https://www.linkedin.com" class="fa fa-linkedin" style="padding: 10px; font-size: 24px; background: #0077B5; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.instagram.com" class="fa fa-instagram" style="padding: 10px; font-size: 24px; background: #E1306C; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.youtube.com" class="fa fa-youtube" style="padding: 10px; font-size: 24px; background: #FF0000; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://www.facebook.com" class="fa fa-facebook" style="padding: 10px; font-size: 24px; background: #3b5998; color: white; text-decoration: none; margin: 5px;"></a>
    <a href="https://twitter.com" class="fa fa-twitter" style="padding: 10px; font-size: 24px; background: #1DA1F2; color: white; text-decoration: none; margin: 5px;"></a>
  </div>
</div>
"""

# Display footer
st.markdown(footer, unsafe_allow_html=True)
