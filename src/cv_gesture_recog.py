import cv2
import mediapipe as mp

# Importing the Necessary Components from Mediapipe
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model Path
MODEL_PATH = "/Users/goduguanilhimam/Downloads/gesture_recognizer.task"

# Initialising the Gesture Recognizer options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

# Video Path
VIDEO_PATH = "/Users/goduguanilhimam/Downloads/more_gestures.mov"

# Creating the OpenCV video capture object
capture = cv2.VideoCapture(VIDEO_PATH)

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

        # Recognizing the Gestures in the Frame
        recognition_result = recognizer.recognize_for_video(
            mediapipe_frame, timestamp_ms=int(capture.get(cv2.CAP_PROP_POS_MSEC))
        )

        # Updating the Detected Gestures onto the frames
        if recognition_result.gestures:
            for gesture in recognition_result.gestures[0]:
                text = f"{gesture.category_name}: {gesture.score:.2f}"
                cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        # Displaying the Frame with the Gesture Updated
        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()
