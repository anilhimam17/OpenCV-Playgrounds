import numpy as np
import cv2
from sys import argv

from numpy._core.numerictypes import int32


def detect_corners_img(path: str) -> None:
    # Reading an Image
    img = cv2.imread(path)

    # Resizing the Image
    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    """
    Conventionally all Edge Detection algorithms work with a grayscale image
    for better accuracy and easier detection of the edges.
    """
    img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    """
    Corner detection is provided in OpenCV by the goodFeaturesToTrack(
        image: MatLike OpenCV Image Object,
        n-corners: OpenCV searches for the best n-corners in the image,
        confidence: A confidence of the algorithm for the n-corners it finds (0 [Not Sure] - 1 [Completely Sure]),
        euclidean-dist: The minimum euclidean distance between two detected edges to not overfit to an edge
    )

    Returns: A list of lists containing the x-y coordinates for each of the corners detected.

    The detected corners are drawn seperately.
    """
    corners = cv2.goodFeaturesToTrack(img_grayscale, 100, 0.5, 10)
    corners = np.array(corners, dtype=int32)
    n_corners = corners.shape[0]

    # Drawing the Corners on the Image
    for corner in corners:
        x, y = corner.ravel()
        img_resized = cv2.circle(img_resized, (x, y), 10, (255, 0, 0), 2)

    # Displaying the total no of corners detected
    font_face = cv2.FONT_HERSHEY_COMPLEX
    text = f"Corners Detected: {n_corners}"
    img_resized = cv2.putText(
        img_resized, text=text, org=(30, 100), fontFace=font_face, fontScale=3,
        thickness=5, color=(255, 0, 255), lineType=cv2.LINE_AA
    )

    # Displaying the Image
    cv2.imshow("Edge Detected Image", img_resized)
    _ = cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_corners_video() -> None:
    # Capturing Livestream video feed from the Camera
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Converting the Livestream Frames to Gray Scale
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Searching for corners in the stream
        corners = cv2.goodFeaturesToTrack(frame_gray, 100, 0.5, 10)
        corners = np.array(corners, dtype=int32)
        n_corners = corners.shape[0]

        # Drawing the Corners
        for corner in corners:
            x, y = corner.ravel()
            frame_resized = cv2.circle(frame_resized, (x, y), 10, (255, 0, 0), 10)

        # Displaying the total no of corners detected
        font_face = cv2.FONT_HERSHEY_COMPLEX
        text = f"Corners Detected: {n_corners}"
        frame_resized = cv2.putText(
            frame_resized, text=text, org=(30, 100), fontFace=font_face, fontScale=1,
            thickness=1, color=(255, 0, 255), lineType=cv2.LINE_AA
        )

        # Displaying the Livestream
        cv2.imshow("Edge Detection Video", frame_resized)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


# Driver Function
def main() -> None:
    # Parsing an Image if passed as a CLI arg
    if len(argv) > 1:
        detect_corners_img(argv[1])
    # Reading a video stream if no args are passed
    else:
        detect_corners_video()


if __name__ == "__main__":
    main()
