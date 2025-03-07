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

# Load YOLO model
model = YOLO("helmet_detector.pt")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Simulate & Detect", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Main detection UI
detect, model_info = st.tabs(['Detection', 'Model Information'])

with detect:
    # Placeholder for clearing content
    stframe = st.empty()  

    if uploaded_file is None:
        # Show title and default video when no file is uploaded
        st.markdown("<h1>Vision Meets<span style='color:#4CAF50;'> SAFETY</span></h1>", unsafe_allow_html=True)
        st.video("assets/vid.mp4")

    else:
        # Clear previous content
        stframe.empty()

        file_type = uploaded_file.type.split("/")[0]  # Get file type (image/video)

        if file_type == "image":
            image = Image.open(uploaded_file)
            image = np.array(image)

            with st.spinner("Processing image..."):
                results = model(image, verbose=False)  # Run YOLO inference
                detected_img = results[0].plot(conf=True)
                detected_img = cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

            stframe.image(detected_img, caption="Detected Image", use_container_width=True)

        elif file_type == "video":
            # Save video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()  # Ensure the file is closed before opening

            # Open video file
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Exit loop when video ends

                # Run YOLO on frame
                results = model(frame, verbose=False)
                detected_frame = results[0].plot(conf=True)

                # Convert frame color format
                detected_frame = cv2.cvtColor(np.array(detected_frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)

                # Display video frame
                stframe.image(detected_frame, channels="BGR", use_container_width=True)

            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()

            # Ensure Windows releases the file before deleting
            time.sleep(1)  
            os.remove(tfile.name)
            
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
