import cv2
from cv2.typing import MatLike
import mediapipe as mp
import time

# Importing the Necessary Components from Mediapipe
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model Path
MODEL_PATH = "/Users/goduguanilhimam/Downloads/gesture_recognizer.task"

# Variable to store the recognized gestures asynchronously
recognized_gesture = None


def gesture_callback(result, output_img: MatLike, timestamp_ms: int) -> None:
    global recognized_gesture
    if result.gestures:
        recognized_gesture = f"{result.gestures[0][0].category_name}: {result.gestures[0][0].score:.2f}"
    else:
        recognized_gesture = None


# Initialising the Gesture Recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_callback
)

# Creating the OpenCV video capture object
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error")
    exit()

# Creating an instance of the Gesture Recognizer to begin working
with GestureRecognizer.create_from_options(options) as recognizer:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("End of Video")
            break

        # Converting the Color Channels for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converting the frame to a MediaPipe image
        mediapipe_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )

        # Updating the Timestamp for real-time video
        timestamp_ms = int(time.time() * 1000)

        # Recognizing the Gestures in the Frame
        recognizer.recognize_async(mediapipe_frame, timestamp_ms=timestamp_ms)

        # Updating the Detected Gestures onto the frames
        if recognized_gesture:
            cv2.putText(frame, recognized_gesture, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        # Displaying the Frame with the Gesture Updated
        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
