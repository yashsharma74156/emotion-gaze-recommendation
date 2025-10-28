import os
import cv2
import dlib
import time
import numpy as np

class GazeDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

        # Final path to .dat file – placed at root: D:\eye_gaze_project\
        base_dir = os.path.dirname(os.path.abspath(__file__))
        predictor_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'shape_predictor_68_face_landmarks.dat'))
        predictor_path = os.path.normpath(predictor_path)

        if not os.path.isfile(predictor_path):
            raise FileNotFoundError(f"shape_predictor_68_face_landmarks.dat NOT found at: {predictor_path}")

        self.predictor = dlib.shape_predictor(predictor_path)
        self.LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
        self.ratio_history = []

        self.last_direction = "Looking Center"
        self.direction_start_time = time.time()
        self.direction_duration = 0

    def get_eye_region(self, landmarks, eye_points):
        return [(landmarks.part(point).x, landmarks.part(point).y) for point in eye_points]

    def get_pupil_position(self, eye_region, gray_frame):
        x_coords = [p[0] for p in eye_region]
        y_coords = [p[1] for p in eye_region]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        eye_img = gray_frame[y_min:y_max, x_min:x_max]
        if eye_img.size == 0:
            return None, None

        thresh = cv2.adaptiveThreshold(eye_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                return cx, eye_img.shape[1]
        return None, None

    def get_smoothed_ratio(self, ratio):
        ratio = float(ratio)
        self.ratio_history.append(ratio)
        if len(self.ratio_history) > 5:
            self.ratio_history.pop(0)
        return sum(self.ratio_history) / len(self.ratio_history)

    def get_gaze_direction(self, pupil_x, eye_width):
        if pupil_x is None or eye_width is None:
            return "Looking Center"

        ratio = pupil_x / eye_width
        smoothed_ratio = self.get_smoothed_ratio(ratio)

        if smoothed_ratio < 0.3:
            return "Looking Left"
        elif smoothed_ratio > 0.7:
            return "Looking Right"
        else:
            return "Looking Center"

    def detect_gaze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye = self.get_eye_region(landmarks, self.LEFT_EYE_POINTS)
            pupil_x, eye_width = self.get_pupil_position(left_eye, gray)
            current_direction = self.get_gaze_direction(pupil_x, eye_width)

            if current_direction == self.last_direction:
                self.direction_duration = time.time() - self.direction_start_time
            else:
                self.last_direction = current_direction
                self.direction_start_time = time.time()
                self.direction_duration = 0

            return current_direction, round(self.direction_duration, 1)

        return "No Face", 0.0
