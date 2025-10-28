# gaze_emotion/emotion_detector.py

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # eye_gaze_project folder path
        model_path = os.path.join(base_dir, 'models', 'emotion_model.h5')
        
        self.model = load_model(model_path)
        self.class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        label = "No Face"  # Default label

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            preds = self.model.predict(roi_gray, verbose=0)
            label = self.class_labels[np.argmax(preds)]
            break  # Only process the first face

        return label
