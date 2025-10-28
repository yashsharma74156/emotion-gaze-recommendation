import cv2
from collections import deque, Counter
from threading import Thread

from Main.gaze_emotion.gaze_detector import GazeDetector
from Main.gaze_emotion.emotion_detector import EmotionDetector

# Global variables to store latest values
latest_gaze = "center"
latest_emotion = "neutral"
is_detection_running = False

def start_detection_background():
    global latest_gaze, latest_emotion, is_detection_running
    if is_detection_running:
        return

    is_detection_running = True
    gaze_detector = GazeDetector()
    emotion_detector = EmotionDetector()

    gaze_window = deque(maxlen=15)
    emotion_window = deque(maxlen=15)

    webcam = cv2.VideoCapture(0)

    while True:
        ret, frame = webcam.read()
        if not ret:
            break

        raw_gaze, _ = gaze_detector.detect_gaze(frame)
        raw_emotion = emotion_detector.detect_emotion(frame)

        if raw_gaze != "No Face" and raw_emotion:
            gaze_window.append(raw_gaze)
            emotion_window.append(raw_emotion)

        if gaze_window:
            latest_gaze = Counter(gaze_window).most_common(1)[0][0].lower()
        if emotion_window:
            latest_emotion = Counter(emotion_window).most_common(1)[0][0].lower()

        cv2.imshow("Capturing Gaze + Emotion", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    webcam.release()
    cv2.destroyAllWindows()
    is_detection_running = False

def start_detection_thread():
    t = Thread(target=start_detection_background, daemon=True)
    t.start()

def get_latest_gaze_emotion():
    return latest_gaze, latest_emotion
