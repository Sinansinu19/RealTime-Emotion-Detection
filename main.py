import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 1. Load the completed Keras 3 format model
try:
    model = load_model('emotion_model.keras')
    print("✅ Model loaded successfully from emotion_model.keras")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure 'emotion_model.keras' is in the exact same directory as this script.")

# 2. Load Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Alphabetical label list matching the 5 directory folders found during training
labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'] 

# Initialize Web Camera with Performance Configuration Backend (DirectShow for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Prevent frame buffer stacking lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    # Fallback to standard initialization if DirectShow fails
    cap = cv2.VideoCapture(0)

# FIX: Initialize the missing frame counter variable outside the loop to prevent NameError
frame_count = 0

# Variables to store the last predicted data for skipped frames
last_label = ""
last_confidence = 0.0
face_detected = False

print("Starting real-time tracking... Press 'q' on your keyboard to exit.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # FIX: Increment the frame counter variable on every single loop iteration
    frame_count += 1

    # Convert to grayscale for Haar Cascade and the Neural Network
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        face_detected = False

    for (x, y, w, h) in faces:
        face_detected = True
        # Draw bounding box around detected faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # FIX: Only run the heavy model prediction on every 3rd frame to stop CPU freezing
        if frame_count % 3 == 0:
            # Extract, crop, and preprocess the region of interest (the face) to strict 48x48x1 grayscale
            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)   # Keeps lighting inputs standardized
            face = cv2.resize(face, (48, 48))
            
            # Reshape to match the exact input dimensions expected by the network: (1, 48, 48, 1)
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Inference
            prediction = model.predict(face, verbose=0)
            predicted_class = np.argmax(prediction)
            
            # Guard against index errors and cache the values
            if predicted_class < len(labels):
                last_label = labels[predicted_class]
                last_confidence = prediction[0][predicted_class] * 100
            else:
                last_label = "Unknown"
                last_confidence = 0.0

        # Render the text (either newly predicted or from cached frame history)
        if last_label != "":
            display_text = f"{last_label} ({last_confidence:.1f}%)"
            cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    # Render video stream UI window
    cv2.imshow('Real-Time Emotion Detector', frame)

    # Break loop safely with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()