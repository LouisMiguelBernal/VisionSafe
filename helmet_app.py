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

# 📸 Image Upload Mode
if mode == "📸 Image Detection":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Progress bar
        with st.spinner("Processing image..."):
            results = model(image, verbose=False)  # Run YOLO inference
            detected_img = results[0].plot(conf=True)
            detected_img = cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        st.image(detected_img, caption="Detected Image", use_container_width=True)

# 🎥 Real-Time Webcam Mode using WebRTC
elif mode == "🎥 Real-Time Webcam":
    st.write("⚡ Turn on your camera for real-time helmet detection.")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model  # Load model once

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Resize frame for better performance
            img = cv2.resize(img, (640, 480))

            try:
                results = self.model(img, verbose=False)

                if len(results[0].boxes) == 0:
                    return img  # Return original frame if no detection

                detected_img = results[0].plot(conf=True)
                detected_img = cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                return detected_img
            except Exception as e:
                st.error(f"Error processing frame: {e}")
                return img  # Return original frame if error

    # WebRTC Streamer with optimized settings
    webrtc_streamer(
        key="helmet-detection",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"width": 640, "height": 480, "facingMode": "user"}, "audio": False},
    )
