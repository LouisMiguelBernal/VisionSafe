import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

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

# Real-Time Webcam Mode
elif mode == "🎥 Real-Time Webcam":
    st.write("⚡ Turn on your camera for real-time helmet detection.")

    if not hasattr(cv2, "VideoCapture"):
        st.error("Webcam access is not supported in this environment.")
    else:
        run_webcam = st.checkbox("Start Webcam")

        if run_webcam:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break

                # Run YOLO inference
                results = model(frame)
                detected_img = results[0].plot()

                # Convert to RGB for Streamlit
                detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

                # Display the video stream
                stframe.image(detected_img, channels="RGB", use_column_width=True)

            cap.release()

        
