import cv2
import mediapipe as mp
import time
import joblib
import numpy as np
from cv2.typing import MatLike

# Load trained classifier and scaler
clf = joblib.load("./models/gesture_class_svm.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Setting up MediaPipe HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "./models/hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

capture = cv2.VideoCapture(0)
last_timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        rgb_frame: MatLike = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mediapipe_frame = mp.Image(mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1
        last_timestamp = timestamp_ms

        landmark_result = landmarker.detect_for_video(mediapipe_frame, timestamp_ms=timestamp_ms)

        gesture_text = ""
        if landmark_result and landmark_result.hand_landmarks:
            hand = landmark_result.hand_landmarks[0]
            # Flatten landmarks into a feature vector
            landmarks_flat = [val for landmark in hand for val in (landmark.x, landmark.y, landmark.z)]
            # Preprocess the features
            features = np.array(landmarks_flat).reshape(1, -1)
            features = scaler.transform(features)
            # Predict custom gesture
            prediction = clf.predict(features)[0]
            gesture_text = f"Custom Gesture: {prediction}"

            # Optionally, draw landmarks on frame
            h, w, _ = frame.shape
            for landmark in hand:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.putText(frame, gesture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Custom Gesture Recognition", frame)
        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
