import numpy as np
import cv2

# Video Capture resource
capture = cv2.VideoCapture(0)


def display_hsv_stream() -> None:
    while True:
        # Reading the Stream of Frames live from the Camera
        ret, frame = capture.read()

        # Converting the BGR image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Displaying the Image
        cv2.imshow("Image in HSV", hsv)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


def extract_color_from_stream() -> None:
    while True:
        # Retrieving the Frames in real-time
        ret, frame = capture.read()

        # Converting the Frames to HSV for color extraction
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # upper bound and lower bound
        lb = np.array([90, 50, 50])
        ub = np.array([130, 255, 255])

        # Creating a mask to pick up the colors
        mask = cv2.inRange(hsv_frame, lb, ub)

        # Applying the Mask and updating the image
        mask_out = cv2.bitwise_and(frame, frame, mask=mask)

        # Displaying the Image
        cv2.imshow("Filtered Image", mask_out)
        cv2.imshow("Mask", mask)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


extract_color_from_stream()
