import sys
import os
import cv2
from collections import deque, Counter

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from gaze_emotion.gaze_detector import GazeDetector
from gaze_emotion.emotion_detector import EmotionDetector
from recommender.product_recommender import ProductRecommender

# Initialize components
gaze_detector = GazeDetector()
emotion_detector = EmotionDetector()
recommender = ProductRecommender()

# Stabilization buffers (sliding window)
gaze_window = deque(maxlen=15)       # For smoothing gaze direction
emotion_window = deque(maxlen=15)    # For smoothing emotion

last_stable_gaze = None
last_stable_emotion = None
last_recommendation = []

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Get current gaze + duration
    raw_gaze, duration = gaze_detector.detect_gaze(frame)
    gaze_window.append(raw_gaze)

    # Get current emotion
    raw_emotion = emotion_detector.detect_emotion(frame)
    emotion_window.append(raw_emotion)

    # Stabilize gaze and emotion using majority vote
    stable_gaze = Counter(gaze_window).most_common(1)[0][0]
    stable_emotion = Counter(emotion_window).most_common(1)[0][0]

    # Only update recommendations if BOTH gaze and emotion change
    if stable_gaze != last_stable_gaze or stable_emotion != last_stable_emotion:
        last_recommendation = recommender.recommend(stable_emotion.lower(), stable_gaze.lower())
        last_stable_gaze = stable_gaze
        last_stable_emotion = stable_emotion

    # Print log
    print(f"Gaze: {stable_gaze} ({round(duration, 1)}s) | Emotion: {stable_emotion}")
    print("Recommended Products:")
    for product in last_recommendation:
        print(f"- {product['title']} | ₹{product['price']}")

    # Draw on frame
    cv2.putText(frame, f"Gaze: {stable_gaze} ({round(duration, 1)}s)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Emotion: {stable_emotion}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    y_offset = 120
    for product in last_recommendation[:2]:
        display_text = f"{product['title'][:30]} - ₹{product['price']}"
        cv2.putText(frame, display_text, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        y_offset += 30

    cv2.imshow("AI Recommender | Gaze + Emotion", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

webcam.release()
cv2.destroyAllWindows()
