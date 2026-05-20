import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIGURATION & STYLE
# ============================================================================

st.set_page_config(
    page_title="Real-Time Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        body {
            background-color: #0e1117;
            color: #e5e5e5;
        }
        .css-18ni7ap, .css-12oz5g7, .css-1d391kg, .css-1y4p8pa {
            background-color: #111827 !important;
            color: #e5e5e5 !important;
        }
        .stButton>button {
            background-color: #2563eb !important;
            color: #ffffff !important;
        }
        .stMetric {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }
        div[data-testid="stSidebar"] {
            background-color: #111827 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# MODEL & CASCADE LOADING
# ============================================================================

@st.cache_resource
def load_emotion_model():
    try:
        return load_model('emotion_model.keras')
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Ensure 'emotion_model.keras' is in the same directory as this script.")
        return None

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_emotion_model()
face_cascade = load_face_cascade()

EMOTION_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': (255, 0, 0),
    'Happy': (0, 255, 0),
    'Neutral': (0, 0, 255),
    'Sad': (255, 165, 0),
    'Surprise': (255, 0, 255)
}

# ============================================================================
# VIDEO TRANSFORMER
# ============================================================================

class IntelligentEmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_prediction = "Scanning..."
        self.confidence_score = 0.0
        self.frame_count = 0
        self.emotion_history = []
        self.confidence_history = []
        self.timestamps = []

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if self.frame_count % 3 == 0 and len(faces) > 0:
                for (x, y, w, h) in faces:
                    y_start, y_end = max(0, y), min(gray.shape[0], y + h)
                    x_start, x_end = max(0, x), min(gray.shape[1], x + w)

                    face = gray[y_start:y_end, x_start:x_end]
                    if face.size == 0:
                        continue

                    face = cv2.equalizeHist(face)
                    face = cv2.resize(face, (48, 48))
                    face = np.expand_dims(face, axis=0)
                    face = np.expand_dims(face, axis=-1)

                    prediction = model(face, training=False).numpy()
                    predicted_class = np.argmax(prediction)
                    confidence = float(prediction[0][predicted_class] * 100)

                    if predicted_class < len(EMOTION_LABELS):
                        self.last_prediction = EMOTION_LABELS[predicted_class]
                        self.confidence_score = confidence
                    else:
                        self.last_prediction = "Unknown"
                        self.confidence_score = 0.0

                    self.emotion_history.append(self.last_prediction)
                    self.confidence_history.append(self.confidence_score)
                    self.timestamps.append(datetime.now())
                    break

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (236, 141, 18), 2)
                cv2.putText(
                    img,
                    f"{self.last_prediction} ({self.confidence_score:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (36, 255, 12),
                    2,
                )
        except Exception as e:
            print(f"Internal WebRTC Processing Bypass: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================================
# UI LAYOUT
# ============================================================================

st.title(" Real-Time Emotion Detection System")
st.markdown("**Production-ready Streamlit frontend with real-time analytics powered by Keras 3**")

with st.sidebar:
    st.header("⚙️ Configuration")

    if model is None:
        st.error("Model not loaded. Please check the model file.")
    else:
        st.success("✅ Model loaded successfully")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    mode_selection = st.radio(
        "Select Mode:",
        ("Live Streaming", "Single Frame Analysis"),
        help="Choose between continuous streaming or single frame analysis"
    )

    st.markdown("---")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_stats = st.checkbox("Show Real-time Statistics", value=True)

if model is None:
    st.stop()

# ============================================================================
# SINGLE FRAME UTILITY
# ============================================================================

def preprocess_single_face(face_roi):
    face = cv2.equalizeHist(face_roi)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    predictions = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[max(0, y):min(gray.shape[0], y + h), max(0, x):min(gray.shape[1], x + w)]
        if face.size == 0:
            continue
        face_preprocessed = preprocess_single_face(face)
        prediction = model(face_preprocessed, training=False).numpy()
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class] * 100)
        if predicted_class < len(EMOTION_LABELS):
            label = EMOTION_LABELS[predicted_class]
            color = EMOTION_COLORS[label]
            cv2.putText(
                img,
                f"{label} ({confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            predictions.append({'emotion': label, 'confidence': confidence})

    return img, predictions

# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🎥 Live Webcam Stream")
    ctx = webrtc_streamer(
        key="emotion-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
        video_transformer_factory=IntelligentEmotionTransformer,
    )

with col2:
    st.subheader("📈 Live Telemetry Metrics")
    local_emotions = []
    local_confidences = []
    local_times = []
    current_emotion = "Scanning..."
    current_confidence = 0.0

    if ctx.video_transformer is not None:
        local_emotions = list(ctx.video_transformer.emotion_history)
        local_confidences = list(ctx.video_transformer.confidence_history)
        local_times = list(ctx.video_transformer.timestamps)
        current_emotion = ctx.video_transformer.last_prediction
        current_confidence = ctx.video_transformer.confidence_score

    st.metric("Dominant Emotion", current_emotion)
    st.metric("Confidence", f"{current_confidence:.1f}%")

    if show_stats:
        st.markdown("---")
        st.subheader("📊 Live Analytics")

        emotion_counts = {}
        for emotion in local_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        for emotion, count in emotion_counts.items():
            percentage = (count / len(local_emotions)) * 100 if local_emotions else 0
            if emotion == 'Angry':
                col_a.metric("😠 Angry", f"{percentage:.1f}%", f"{count} hits")
            elif emotion == 'Happy':
                col_b.metric("😊 Happy", f"{percentage:.1f}%", f"{count} hits")
            elif emotion == 'Neutral':
                col_c.metric("😐 Neutral", f"{percentage:.1f}%", f"{count} hits")
            elif emotion == 'Sad':
                col_d.metric("😢 Sad", f"{percentage:.1f}%", f"{count} hits")
            elif emotion == 'Surprise':
                col_e.metric("😲 Surprise", f"{percentage:.1f}%", f"{count} hits")

        if local_confidences:
            st.markdown("---")
            st.subheader("Confidence Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=local_confidences,
                mode='lines+markers',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4),
            ))
            fig.update_layout(
                title='Confidence Trend',
                xaxis_title='Inference Samples',
                yaxis_title='Confidence (%)',
                template='plotly_dark',
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("📸 Single Frame Analysis")
uploaded_file = st.file_uploader("Upload an image or capture from webcam:", type=["jpg", "jpeg", "png", "bmp"])
use_camera = st.checkbox("Use webcam instead of upload")
img = None

if use_camera:
    picture = st.camera_input("Take a picture")
    if picture is not None:
        file_bytes = np.frombuffer(picture.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if img is not None:
    processed_img, predictions = process_image(img)
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
    with col2:
        st.subheader("Detected Emotions")
        if predictions:
            for pred in predictions:
                emotion = pred['emotion']
                confidence = pred['confidence']
                st.write(f"**{emotion}** — {confidence:.1f}%")
        else:
            st.warning("No faces detected in the image.")
