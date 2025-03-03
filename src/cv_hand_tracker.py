import cv2
from cv2.typing import MatLike
import mediapipe as mp
import time


# Setting up the components for MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmaker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model Path
MODEL_PATH = "./models/hand_landmarker.task"

# Initialising the Hand Land Marker Model options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Video Path
VIDEO_PATH = "/Users/goduguanilhimam/Downloads/hand_landmarker.mov"

# OpenCV Capture Channel
capture = cv2.VideoCapture(VIDEO_PATH)

# Control Variable to ensure the increasing count of timestamps
last_timestamp = 0

# Creating an instance of the HandLandmarker and running it in Video Mode
with HandLandmaker.create_from_options(options) as landmarker:
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("End of the Video")
            break

        # Converting the BGR frames to RGB for MediaPipe processing
        rgb_frame: MatLike = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converting the frame to a MediaPipe frame
        mediapipe_frame = mp.Image(
            mp.ImageFormat.SRGB, data=rgb_frame
        )

        # Generating a self increasing timestamp
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1
        last_timestamp = timestamp_ms

        # Landmark Detection from video
        landmark_result = landmarker.detect_for_video(
            mediapipe_frame, timestamp_ms=timestamp_ms
        )

        # Drawing the result onto the video
        if (
            landmark_result and 
            landmark_result.hand_landmarks  # .hand_landmarks stores the list of landmarks for each hand
        ):
            for hand in landmark_result.hand_landmarks:
                for landmark in hand:
                    # Extracting the height and width of the frames
                    h, w, _ = frame.shape

                    # Remapping the values of the landmarks to match the height and width
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    # Drawing the circles onto the landmarks that were detected in the live stream
                    frame = cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        cv2.imshow("Handlandmark Video Detector", frame)

        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
