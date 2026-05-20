# 🚀 Production-Ready Streamlit Frontend Setup Guide

## Overview
This guide provides step-by-step instructions to set up and run a production-ready Streamlit web application with real-time emotion detection using your trained Keras 3 model. The application uses **streamlit-webrtc** for browser-based webcam streaming instead of OpenCV windows.

---

## ✅ Prerequisites

Before starting, ensure you have:
- **Python 3.8+** installed
- Your trained Keras emotion detection model (`emotion_model.keras`) in the project directory
- Your dataset structure intact (for reference)

---

## 📦 Installation Guide

### Step 1: Create a Virtual Environment (Recommended)

It's best practice to use a virtual environment to isolate project dependencies.

#### On Windows (PowerShell):
```powershell
# Navigate to your project directory
cd "C:\Users\Sinan\Documents\AI"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

#### On macOS/Linux:
```bash
cd ~/Documents/AI
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Required Packages

Run the following command to install all necessary packages:

```bash
pip install streamlit streamlit-webrtc opencv-python-headless tensorflow keras plotly
```

**Package Details:**
- **streamlit**: Web app framework for building the interface
- **streamlit-webrtc**: Real-time browser-based WebRTC streaming for webcam
- **opencv-python-headless**: OpenCV without GUI (no X11 dependency; lightweight)
- **tensorflow**: Deep learning framework
- **keras**: Neural network API
- **plotly**: Interactive data visualization for real-time analytics

### Step 3: Verify Installation

```bash
pip list | findstr /I "streamlit tensorflow keras opencv plotly"
```

You should see all packages listed with their versions.

---

## 🎯 Running the Streamlit Application

### Method 1: Standard Execution (Recommended)

```bash
streamlit run streamlit_app.py
```

**What happens:**
1. Streamlit starts a local development server
2. Opens your default browser automatically (usually at `http://localhost:8501`)
3. The web interface is ready for use

### Method 2: Specify Host and Port

```bash
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
```

### Method 3: Run with Custom Configuration

Create a `.streamlit/config.toml` file in your project directory:

```toml
[server]
port = 8501
headless = false
runOnSave = true

[logger]
level = "info"

[client]
showErrorDetails = true
```

Then run:
```bash
streamlit run streamlit_app.py
```

---

## 🎮 Using the Web Application

### Live Streaming Mode

1. **Start Streaming:**
   - Select "Live Streaming" from the sidebar
   - Click the **"Start"** button to activate your webcam
   - Grant browser permission to access your camera when prompted

2. **Real-time Analysis:**
   - Detected faces are highlighted with bounding boxes
   - Emotion predictions appear above each face with confidence scores
   - Analytics update in real-time below the video stream

3. **View Analytics:**
   - **Emotion Distribution**: Pie chart showing percentage breakdown
   - **Confidence Trend**: Line chart tracking prediction confidence over time
   - **Key Metrics**: Total frames, average confidence, most detected emotion

4. **Stop Streaming:**
   - Click **"Stop"** to end the session and clear the stream

### Single Frame Analysis Mode

1. **Choose Input:**
   - Use the **upload button** to select an image file, OR
   - Check **"Use webcam instead"** to capture a single photo

2. **View Results:**
   - Processed image with detected faces and emotion labels appears on the left
   - Detailed predictions with confidence bars on the right
   - No real-time streaming; single frame analysis only

---

## 📊 Features Explained

### Real-time Emotion Detection
- **Model**: Keras 3 CNN trained on 5 emotions (Angry, Happy, Neutral, Sad, Surprise)
- **Input**: 48×48 grayscale face images
- **Processing**: Haar Cascade for face detection + Histogram Equalization for lighting normalization

### Analytics Dashboard
- **Emotion Distribution Chart**: Visual breakdown of detected emotions
- **Confidence Trend**: Monitor how confident the model is over time
- **Frame Counter**: Track total frames processed during session
- **Average Confidence**: Statistical measure of model reliability

### Browser-based Streaming
- **Technology**: WebRTC for peer-to-peer video streaming
- **Advantage**: No OpenCV windows; smooth streaming in browser
- **Compatibility**: Works on Windows, macOS, Linux, and most modern browsers

---

## 🔧 Troubleshooting

### Issue: "streamlit not found" or "ModuleNotFoundError"
**Solution:**
```bash
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Reinstall packages
pip install streamlit streamlit-webrtc opencv-python-headless tensorflow keras plotly
```

### Issue: Webcam not working or permission denied
**Solution:**
1. Check browser permissions (camera access)
2. Ensure no other application is using the webcam
3. Try a different browser (Chrome/Edge recommended)
4. Restart the Streamlit app and grant permissions again

### Issue: Model file not found (Error loading model)
**Solution:**
- Ensure `emotion_model.keras` is in the same directory as `streamlit_app.py`
- Check file permissions: `ls -la emotion_model.keras` (macOS/Linux) or `dir emotion_model.keras` (Windows)

### Issue: Slow performance or high latency
**Solution:**
1. Reduce frame rate in browser (built-in Streamlit optimization)
2. Close other resource-intensive applications
3. Ensure stable internet connection (for WebRTC)
4. Try: `streamlit run streamlit_app.py --logger.level=warning`

### Issue: Port 8501 already in use
**Solution:**
```bash
# Use a different port
streamlit run streamlit_app.py --server.port=8502
```

---

## 📝 Project Structure

```
AI/
├── streamlit_app.py           # Main Streamlit web application
├── emotion_model.keras         # Trained Keras 3 model
├── main.py                     # Original OpenCV-based script
├── train_model.py              # Model training script
├── convert.py                  # Image format converter
├── SETUP_GUIDE.md              # This file
├── requirements.txt            # Python dependencies (optional)
└── dataset/
    ├── Angry/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprise/
```

---

## 🚀 Production Deployment (Optional)

### Deploy to Streamlit Cloud (Free)

1. Push your project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create new app → Select your repository
4. Streamlit Cloud handles hosting and auto-deploys on git push

**Note:** Webcam access requires HTTPS and browser permissions; local deployment recommended for development.

### Deploy Locally with Custom Server

```bash
# Use Gunicorn or similar for production
streamlit run streamlit_app.py --logger.level=warning --server.enableCORS=false
```

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **streamlit-webrtc**: https://github.com/aiortc/streamlit-webrtc
- **Keras Documentation**: https://keras.io
- **OpenCV Haar Cascades**: https://github.com/opencv/opencv/tree/master/data/haarcascades

---

## 🎓 Next Steps

1. ✅ Run the Streamlit app successfully
2. ✅ Test with your trained model
3. ✅ Customize colors, labels, or analytics
4. ✅ Optimize model performance if needed
5. ✅ Deploy to production (optional)

---

## 💡 Tips for Best Results

- **Lighting**: Ensure adequate lighting for face detection
- **Distance**: Keep your face 12-24 inches from the camera
- **Performance**: Use Chrome or Edge browsers for best WebRTC performance
- **Model Accuracy**: If accuracy is low, consider retraining with more data

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review Streamlit and streamlit-webrtc documentation
3. Check your model's training logs in `train_model.py`

---

**Happy detecting! 😊**
