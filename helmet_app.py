import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load YOLO model
model = YOLO("helmet_detector.pt")

# Streamlit UI
st.title("🛵 Helmet Detection System")

# Sidebar menu
mode = st.sidebar.radio("Choose mode", ["📸 Image Detection", "🎥 Real-Time Webcam"])

# Image Upload Mode
if mode == "📸 Image Detection":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Convert file to OpenCV format
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Run YOLO inference
        results = model(image)
        detected_img = results[0].plot()  # Draw detected boxes

        # Display the results
        st.image(detected_img, caption="Detected Image", use_column_width=True)

# Real-Time Webcam Mode using WebRTC
elif mode == "🎥 Real-Time Webcam":
    st.write("⚡ Turn on your camera for real-time helmet detection.")

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO inference
            results = model(img)
            detected_img = results[0].plot()

            return detected_img

    webrtc_streamer(
        key="helmet-detection",
        video_transformer_factory=VideoTransformer
    )
