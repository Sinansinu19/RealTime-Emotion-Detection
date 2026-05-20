# 🚀 Quick Start - Run in 3 Steps

## Step 1: Install Dependencies

Choose your platform:

### Windows (PowerShell)
```powershell
# Automatic setup (recommended)
.\setup.bat

# OR manual installation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux
```bash
# Automatic setup (recommended)
chmod +x setup.sh
./setup.sh

# OR manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 2: Activate Virtual Environment

### Windows:
```powershell
.\venv\Scripts\Activate.ps1
```

### macOS/Linux:
```bash
source venv/bin/activate
```

---

## Step 3: Run the Application

```bash
streamlit run streamlit_app.py
```

✅ Your browser will automatically open to `http://localhost:8501`

---

## 🎮 Usage

### Live Streaming Mode (Default)
- Click **"Start"** to activate webcam
- Grant browser camera permissions
- Watch real-time emotion detection
- Analytics update in real-time
- Click **"Stop"** to end session

### Single Frame Analysis
- Switch to "Single Frame Analysis" in sidebar
- Upload image or capture from webcam
- View emotions detected with confidence scores

---

## 📦 What Gets Installed

```
streamlit               → Web app framework
streamlit-webrtc       → Browser-based webcam streaming
opencv-python-headless → Computer vision (no GUI)
tensorflow/keras       → Deep learning model
plotly                 → Interactive charts
```

---

## ⚡ Troubleshooting

**Webcam not working?**
- Grant browser camera permissions
- Ensure no other app uses the webcam
- Restart the app

**Port 8501 already in use?**
```bash
streamlit run streamlit_app.py --server.port=8502
```

**ModuleNotFoundError?**
- Check virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

---

## 📊 Features

✅ Real-time emotion detection via browser  
✅ WebRTC streaming (smooth, no OpenCV windows)  
✅ Live analytics dashboard  
✅ Emotion distribution charts  
✅ Confidence trend visualization  
✅ Single image analysis mode  
✅ Production-ready UI  

---

**Ready? Start with Step 1 above!** 😊
