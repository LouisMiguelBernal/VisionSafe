import streamlit as st
import tempfile
import cv2
import numpy as np
import time
from PIL import Image
from ultralytics import YOLO
import os
import warnings
import base64
import tempfile
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="XGE", layout="wide", page_icon='assets/icon.png')

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Convert logo to base64
logo_base64 = get_base64_image('assets/icon.png')

# Display the logo and title using HTML with added margin/padding to move it down
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
    .title {
      font-size: 60px;
      font-family: 'Arial', sans-serif;
      text-align: center;
      margin-bottom: 20px;
    }
    .green {
        color: #4CAF50;  
    }
    .main > div {
        padding-top: 30px;
    }
    .stTabs [role="tablist"] button {
        font-size: 1.2rem;
        padding: 12px 24px;
        margin-right: 10px;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [role="tablist"] button:focus, .stTabs [role="tablist"] button[aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stTabs [role="tabpanel"] {
        padding-top: 30px;
    }
    .logo-and-name {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .logo-img {
        border-radius: 50%;
        width: 50px;
        height: 50px;
    }                
    </style>
    """, unsafe_allow_html=True)
# Main detection UI
detect, model_info = st.tabs(['Detection', 'Model Information'])

with detect:
    # Cache model loading for efficiency
    @st.cache_resource
    def load_model():
        return YOLO("helmet_detector.pt")  # Load YOLO model

    model = load_model()  # Load the model once

    # Sidebar file upload
    uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    # Processing function for images
    def process_image(image):
        image = np.array(image)  # Convert to NumPy array
        results = model(image, verbose=False)  # Run YOLO inference
        detected_img = results[0].plot(conf=True)  # Draw detections
        detected_img = cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        return detected_img

    # Processing function for videos (Optimized)
    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()  # Streamlit frame for video display
        
        # Get video properties
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
        skip_frames = 1  

        while cap.isOpened():
            for _ in range(skip_frames):
                cap.read()  # Skip frames to speed up processing

            ret, frame = cap.read()
            if not ret:
                break  # Exit when video ends

            results = model(frame, verbose=False)  # Run YOLO detection
            detected_frame = results[0].plot(conf=True)  # Draw detections
            detected_frame = cv2.cvtColor(np.array(detected_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)

            stframe.image(detected_frame, channels="BGR", use_container_width=True)

        cap.release()
        cv2.destroyAllWindows()

    # Default video file path
    default_video_path = "assets/vid.mp4"  # Ensure this file exists in the same directory as the script

    # Main UI logic
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

            os.remove(temp_file_path)  # Cleanup temporary file after processing

    # If no file is uploaded, play the default video
    elif os.path.exists(default_video_path):
        st.video(default_video_path)
    else:
        st.warning("No video uploaded, and default video file is missing!")

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
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)
st.markdown(footer, unsafe_allow_html=True)
