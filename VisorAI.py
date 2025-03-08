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
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Streamlit page config
st.set_page_config(page_title="VisorAI", layout="wide", page_icon="assets/icon.png")

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

# Initialize session state for tracking sound and uploads
if "last_play_time" not in st.session_state:
    st.session_state.last_play_time = 0
if "last_detected_classes" not in st.session_state:
    st.session_state.last_detected_classes = set()
if "last_upload_time" not in st.session_state:
    st.session_state.last_upload_time = 0

# Sound mapping
SOUND_FILES = {
    "Child-Pedestrian Crossing": "assets/child_pedestrian_crossing.mp3",
    "Give Way": "assets/give_way.mp3",
    "Speed Limit": "assets/speed_limit.mp3",
    "Stop": "assets/stop.mp3",
}

def generate_audio_js(audio_file):
    """Generate JavaScript to force autoplay audio."""
    if not os.path.exists(audio_file):
        return ""

    # Convert audio to base64
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    return f"""
    <script>
        var audio = new Audio("data:audio/mp3;base64,{audio_base64}");
        audio.play().catch(error => console.log("Autoplay failed:", error));
    </script>
    """

def play_sound(class_name):
    """Triggers JavaScript-based audio playback."""
    audio_file = SOUND_FILES.get(class_name)
    if not audio_file:
        return  

    # Prevent sound spam (minimum delay 1.5s)
    current_time = time.time()
    if current_time - st.session_state.last_play_time < 1.5:
        return  

    st.session_state.last_play_time = current_time
    print(f"🔊 Playing sound for: {class_name}")

    # Inject JavaScript to play sound
    audio_js = generate_audio_js(audio_file)
    st.components.v1.html(audio_js, height=0)

def new_upload_detected():
    """Detects if a new file was uploaded and triggers sound."""
    current_time = time.time()
    if current_time - st.session_state.last_upload_time >= 0.5:
        st.session_state.last_upload_time = current_time
        return True
    return False

# Load YOLO model
@st.cache_resource
def load_model():
    model_path = "assets/visor.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    print("✅ YOLO Model Loaded")
    return YOLO(model_path)

model = load_model()
if model is None:
    st.stop()

# Streamlit UI Tabs
detect, model_info = st.tabs(["Detection", "Model Information"])

def check_detections(results):
    """Triggers sound only for newly detected classes."""
    detected_classes = set()
    
    for detection in results[0].boxes:
        confidence = detection.conf[0].item()
        class_index = int(detection.cls[0].item())
        class_name = model.names[class_index]

        print(f"📸 Detected: {class_name}, Confidence: {confidence}")

        if confidence > 0.2 and class_name in SOUND_FILES:
            detected_classes.add(class_name)

    # Play sound for newly detected classes
    new_detections = detected_classes - st.session_state.last_detected_classes
    for class_name in new_detections:
        play_sound(class_name)

    st.session_state.last_detected_classes = detected_classes  # Update session state

def process_image(image):
    """Processes an image for object detection."""
    image = np.array(image)
    
    # Convert RGB image (PIL) to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image, verbose=False)

    check_detections(results)

    detected_img = results[0].plot(conf=True)
    
    # Convert back to RGB for correct display
    return cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_BGR2RGB)  

def process_video(video_path):
    """Processes video frames for object detection."""
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        check_detections(results)

        detected_frame = results[0].plot(conf=True)
        detected_frame = cv2.cvtColor(np.array(detected_frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)

        stframe.image(detected_frame, channels="RGB", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Detection functionality
with detect:
    if uploaded_file:
        # Detect new upload and reset sound tracking
        if new_upload_detected():
            st.session_state.last_detected_classes.clear()

        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            image = Image.open(uploaded_file).convert("RGB")  # Ensure proper color format
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
        st.video("assets/vid.mp4")

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
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set Streamlit page config
st.set_page_config(page_title="VisorAI", layout="wide", page_icon="assets/icon.png")

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

# Initialize session state for tracking sound and uploads
if "last_play_time" not in st.session_state:
    st.session_state.last_play_time = 0
if "last_detected_classes" not in st.session_state:
    st.session_state.last_detected_classes = set()
if "last_upload_time" not in st.session_state:
    st.session_state.last_upload_time = 0

# Sound mapping
SOUND_FILES = {
    "Child-Pedestrian Crossing": "assets/child_pedestrian_crossing.mp3",
    "Give Way": "assets/give_way.mp3",
    "Speed Limit": "assets/speed_limit.mp3",
    "Stop": "assets/stop.mp3",
}

def generate_audio_js(audio_file):
    """Generate JavaScript to force autoplay audio."""
    if not os.path.exists(audio_file):
        return ""

    # Convert audio to base64
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

    return f"""
    <script>
        var audio = new Audio("data:audio/mp3;base64,{audio_base64}");
        audio.play().catch(error => console.log("Autoplay failed:", error));
    </script>
    """

def play_sound(class_name):
    """Triggers JavaScript-based audio playback."""
    audio_file = SOUND_FILES.get(class_name)
    if not audio_file:
        return  

    # Prevent sound spam (minimum delay 1.5s)
    current_time = time.time()
    if current_time - st.session_state.last_play_time < 1.5:
        return  

    st.session_state.last_play_time = current_time
    print(f"🔊 Playing sound for: {class_name}")

    # Inject JavaScript to play sound
    audio_js = generate_audio_js(audio_file)
    st.components.v1.html(audio_js, height=0)

def new_upload_detected():
    """Detects if a new file was uploaded and triggers sound."""
    current_time = time.time()
    if current_time - st.session_state.last_upload_time >= 0.5:
        st.session_state.last_upload_time = current_time
        return True
    return False

# Load YOLO model
@st.cache_resource
def load_model():
    model_path = "assets/visor.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    print("✅ YOLO Model Loaded")
    return YOLO(model_path)

model = load_model()
if model is None:
    st.stop()

# Streamlit UI Tabs
detect, model_info = st.tabs(["Detection", "Model Information"])

def check_detections(results):
    """Triggers sound only for newly detected classes."""
    detected_classes = set()
    
    for detection in results[0].boxes:
        confidence = detection.conf[0].item()
        class_index = int(detection.cls[0].item())
        class_name = model.names[class_index]

        print(f"📸 Detected: {class_name}, Confidence: {confidence}")

        if confidence > 0.2 and class_name in SOUND_FILES:
            detected_classes.add(class_name)

    # Play sound for newly detected classes
    new_detections = detected_classes - st.session_state.last_detected_classes
    for class_name in new_detections:
        play_sound(class_name)

    st.session_state.last_detected_classes = detected_classes  # Update session state

def process_image(image):
    """Processes an image for object detection."""
    image = np.array(image)
    
    # Convert RGB image (PIL) to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = model(image, verbose=False)

    check_detections(results)

    detected_img = results[0].plot(conf=True)
    
    # Convert back to RGB for correct display
    return cv2.cvtColor(np.array(detected_img, dtype=np.uint8), cv2.COLOR_BGR2RGB)  

def process_video(video_path):
    """Processes video frames for object detection."""
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        check_detections(results)

        detected_frame = results[0].plot(conf=True)
        detected_frame = cv2.cvtColor(np.array(detected_frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)

        stframe.image(detected_frame, channels="RGB", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Detection functionality
with detect:
    if uploaded_file:
        # Detect new upload and reset sound tracking
        if new_upload_detected():
            st.session_state.last_detected_classes.clear()

        file_type = uploaded_file.type.split("/")[0]

        if file_type == "image":
            image = Image.open(uploaded_file).convert("RGB")  # Ensure proper color format
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
        st.video("assets/vid.mp4")

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
>>>>>>> ec76a38 (Initial commit with visor.pt)
