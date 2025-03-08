import csv
import cv2
from cv2.typing import MatLike
import mediapipe as mp
import time


# Setting the configurations for the mediapipe model
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model Path
MODEL_PATH = "./models/hand_landmarker.task"

# Initialising the HandLandmarker model with Options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# OpenCV Capture Channel to read livestream
capture = cv2.VideoCapture(0)

# Monotonic Increase of the Timestamps in a Livestream
last_timestamp = 0

# CSV file to keep track of all the hand landmarks being recorded
csv_file = open("./db/landmark_dataset.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
header = ["label"] + [f"p{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]
csv_writer.writerow(header)

# Variable to Flag and Recognized Gesture alike from Unknown
current_label: str | None = None
gesture_ctr: int = 1

# Creating an instance of the Handlandmark Detector and running it in Live Stream Mode
with HandLandmarker.create_from_options(options) as landmarker:

    # Retrieving the frames from the livestream
    while True:
        ret, frame = capture.read()

        if not ret:
            print("Error retrieving livestream")

        # Converting BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converting the RGB frame to MediaPipe image format
        mediapipe_frame = mp.Image(
            mp.ImageFormat.SRGB, data=rgb_frame
        )

        # Generating the Timestamp
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1
            last_timestamp = timestamp_ms

        # Detecting the Landmarks in the Frame
        landmark_result = landmarker.detect_for_video(
            mediapipe_frame, timestamp_ms=timestamp_ms
        )

        if (landmark_result and landmark_result.hand_landmarks):
            for hand in landmark_result.hand_landmarks:
                for landmark in hand:
                    h, w, _ = frame.shape

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    frame: MatLike = cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)

        # Displaying the Livestream
        frame = cv2.putText(frame, f"Gesture Name: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Handlandmark Live Detector", frame)

        # Ending the Livestream
        keyStroke = cv2.waitKey(1)
        if keyStroke == ord("q"):
            print("Ending Livestream")
            break
        elif keyStroke == ord("n"):
            current_label = "New Gesture - " + str(gesture_ctr)
        elif keyStroke == ord("s") and current_label:
            if landmark_result and landmark_result.hand_landmarks:
                hand = landmark_result.hand_landmarks[0]
                landmark_flatten: list[float] = [val for landmark in hand for val in (landmark.x, landmark.y, landmark.z)]
                csv_writer.writerow([gesture_ctr] + list(landmark_flatten))

            # Resetting the Label
            current_label = None
            gesture_ctr += 1

capture.release()
cv2.destroyAllWindows()
