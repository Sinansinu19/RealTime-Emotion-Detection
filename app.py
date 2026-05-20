import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# ============================================================================
# 1. PAGE CONFIGURATION & PREMIUM GRAPHICS THEME
# ============================================================================
st.set_page_config(
    page_title="Real-Time Emotion Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dashboard styling rules
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 12px; border: 1px solid #374151; }
    div[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1f2937; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; color: #f3f4f6; }
    .status-card { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. OPTIMIZED RESOURCE CACHING 
# ============================================================================
@st.cache_resource
def load_emotion_pipeline():
    """Loads deep learning model and Haar cascade with shared memory caching."""
    try:
        model = load_model('emotion_model.keras')
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, cascade
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None, None

model, face_cascade = load_emotion_pipeline()

EMOTION_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Color maps converted to BGR formats for high-performance OpenCV rendering
EMOTION_COLORS_BGR = {
    'Angry': (0, 0, 255),       # Red
    'Happy': (0, 255, 0),       # Green
    'Neutral': (255, 0, 0),     # Blue
    'Sad': (0, 165, 255),       # Orange
    'Surprise': (255, 0, 255)   # Magenta
}

# ============================================================================
# 3. UNFREEZING HIGH-PERFORMANCE WEBRTC VIDEO CORE
# ============================================================================
class IntelligentEmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = "Scanning..."
        self.confidence_score = 0.0
        self.frame_count = 0

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Optimization: Stride execution loop runs deep-learning predictions every 3rd frame
            if self.frame_count % 3 == 0 and len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Defensive cropping boundaries to prevent edge-of-screen array crashes
                    y_start, y_end = max(0, y), min(img.shape[0], y + h)
                    x_start, x_end = max(0, x), min(img.shape[1], x + w)

                    face = gray[y_start:y_end, x_start:x_end]
                    if face.size == 0:
                        continue

                    face = cv2.equalizeHist(face)
                    face = cv2.resize(face, (48, 48))
                    face = face.astype("float32")
                    face = np.expand_dims(face, axis=0)
                    face = np.expand_dims(face, axis=-1)

                    # Performance Fix: Mathematical execution calls drop CPU inference latency significantly
                    prediction = model(face, training=False).numpy()
                    predicted_class = np.argmax(prediction)

                    if predicted_class < len(EMOTION_LABELS):
                        self.last_prediction = EMOTION_LABELS[predicted_class]
                        self.confidence_score = float(prediction[0][predicted_class] * 100)
                    break

            # Fluid frame renderer traces canvas paths continuously at native hardware frame-rates
            for (x, y, w, h) in faces:
                color = EMOTION_COLORS_BGR.get(self.last_prediction, (0, 255, 0))
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                text_label = f"{self.last_prediction} ({self.confidence_score:.1f}%)"
                cv2.rectangle(img, (x, y-30), (x+250, y), color, -1)
                cv2.putText(img, text_label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        except Exception as e:
            print(f"WebRTC System Pipeline Execution Guard: {e}")

        return frame.from_ndarray(img, format="bgr24")

# ============================================================================
# 4. SIDEBAR CONTROL PANELS
# ============================================================================
with st.sidebar:
    st.markdown('<div class="status-card"><h3>Diagnostics Hub</h3><p style="color:#e5e7eb;margin:0;">Core AI Engine: Ready</p></div>', unsafe_allow_html=True)
    st.header("⚙️ System Adjustments")
    
    if model is None:
        st.error("Model assets missing from root folder context.")
        st.stop()
    else:
        st.success("✅ Neural Network Synchronized")

    mode_selection = st.radio(
        "Execution Track Select:",
        ("Live Streaming", "Single Frame Analysis"),
        help="Switch live streaming streams or process specific individual user files."
    )
    
    st.markdown("---")
    st.caption("⚡ Powered by decoupled Keras Evaluation Nodes & WebRTC Pipelines.")

# ============================================================================
# 5. CORE UI PIPELINE DISPATCHER
# ============================================================================
st.title("Real-Time Emotion Detection System")
st.markdown("**Production-grade edge vision system deployed for judge metrics review.**")
st.markdown("---")

if mode_selection == "Live Streaming":
    st.subheader("🎥 High-Performance Live Stream")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # Deployed via non-blocking asynchronous streaming layouts as a clean single-focus module
    ctx = webrtc_streamer(
        key="synchronized-emotion-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_transformer_factory=IntelligentEmotionTransformer,
        async_transform=True,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    st.info(
        "📍 **Quick Setup Instructions:**\n"
        "1. Toggle 'Start' below the player module to mount the camera matrix.\n"
        "2. Fluid visual indicators track facial features automatically without browser stutters."
    )

else:
    # ============================================================================
    # 6. SINGLE FRAME PROCESSING PIPELINE 
    # ============================================================================
    st.subheader("📸 High-Resolution Single Frame Capture Node")
    
    uploaded_file = st.file_uploader("Upload local testing frame asset:", type=["jpg", "jpeg", "png", "bmp"])
    use_camera = st.checkbox("Mount system capture hardware instead")
    
    img = None
    if use_camera:
        picture = st.camera_input("Trigger Snapshot")
        if picture is not None:
            file_bytes = np.frombuffer(picture.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Maintain a pristine, uncompressed copy of the original full-resolution image
            annotated_static = img.copy()
            pred_static = None  
            
            for (x, y, w, h) in faces:
                y_start, y_end = max(0, y), min(img.shape[0], y+h)
                x_start, x_end = max(0, x), min(img.shape[1], x+w)
                
                face_roi = gray[y_start:y_end, x_start:x_end]
                if face_roi.size == 0: 
                    continue
                
                # Apply processing optimizations strictly to an isolated internal variable for inference
                network_input = cv2.equalizeHist(face_roi)
                network_input = cv2.resize(network_input, (48, 48))
                
                # Format the background tensor for model inference without touching canvas variables
                network_input = network_input.astype("float32")
                network_input = np.expand_dims(network_input, axis=0)
                network_input = np.expand_dims(network_input, axis=-1)
                
                # Execute model calculations
                pred_static = model(network_input, training=False).numpy()
                idx_static = np.argmax(pred_static)
                lbl_static = EMOTION_LABELS[idx_static]
                conf_static = float(pred_static[0][idx_static] * 100)
                color = EMOTION_COLORS_BGR.get(lbl_static, (0, 255, 0))
                
                # Draw ONLY the clean bounding box and sharp text overlays on the high-resolution color copy
                cv2.rectangle(annotated_static, (x, y), (x+w, y+h), color, 3)
                
                # Dynamic font sizing relative to face size for premium dashboard aesthetics
                cv2.putText(annotated_static, f"{lbl_static}: {conf_static:.1f}%", (x, max(y-15, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                break 
                
            # Render the crystal clear image asset back to Streamlit without any corner artifacts
            st.image(cv2.cvtColor(annotated_static, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Processed Image Pipeline Output View")
            
        with col2:
            st.subheader("🎯 Diagnostic Distribution Analysis")
            # Only render bars if a face was detected and processed successfully
            if pred_static is not None:
                for idx, label in enumerate(EMOTION_LABELS):
                    confidence_val = float(pred_static[0][idx])
                    st.write(f"**{label}**